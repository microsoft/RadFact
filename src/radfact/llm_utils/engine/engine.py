#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import json
import logging
import multiprocessing
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm import tqdm

from radfact.llm_utils.engine.arguments import LLMAPIArguments, LLMEngineArguments
from radfact.llm_utils.engine.data_subset import DataSubset
from radfact.llm_utils.engine.endpoint_utils import get_endpoints_dict_sorted_by_speed, replicate_same_endpoint_n_times
from radfact.llm_utils.engine.redis_cache import get_redis_cache
from radfact.llm_utils.processor.base_processor import BaseModelWithId, BaseProcessor

logger = logging.getLogger(__name__)
# To avoid excessive logging from azure, azureml and health_azure
for package in ["azure", "azureml", "health_azure"]:
    package_logger = logging.getLogger(package)
    package_logger.setLevel(logging.WARNING)

DictOfShards = dict[str, DataSubset]
OutputsType = list[dict[str, Any]]


def get_subfolder(root: Path, subfolder: str) -> Path:
    """Get the subfolder of the given root folder, creating it if it does not exist.

    :param root: The root folder.
    :param subfolder: The subfolder to create.
    :return: The created subfolder.
    """
    output_folder = root / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


class LLMEngine:
    """Engine for processing data with an LLM api. The dataset is sharded into subsets proportional to the speed factor
    of each endpoint and processed in parallel. The outputs are aggregated at the end into a single json file. The
    progress and skipped IDs are saved to file at the end of processing. The final outputs are saved to the
    final_output_folder if provided, otherwise they are saved to the progress_output_folder. The progress and skipped
    IDs are saved to the run_output_folder. The final output is a json file containing the outputs from all shards and
    is named after the filename specified in the processing config.
    """

    OUTPUT_FILES_PREFIX = "outputs"
    PROGRESS_FILENAME = "progress.csv"
    PROGRESS_FILE_ATTRIBUTE = "progress_file"
    SKIPPED_FILENAME = "skipped.csv"
    SKIPPED_FILE_ATTRIBUTE = "skipped_file"

    def __init__(
        self,
        cfg: DictConfig,
        processor: BaseProcessor[Any, Any],
        dataset_df: pd.DataFrame,
        progress_output_folder: Path,
        final_output_folder: Path | None = None,
        row_to_query_fn: Callable[["pd.Series[Any]"], Any] = lambda x: x,
        verbose: bool = False,
    ) -> None:
        """
        :param cfg: The configuration for the processing pipeline.
        :param processor: A processor object for the structured data, it has to implement set_model and run methods.
        :param dataset_df: The dataframe containing the dataset to be processed.
        :param progress_output_folder: The output folder where intermediate results are saved.
        :param final_output_folder: An optional final output folder where the final results are saved.
        :param row_to_query_fn: A function that formats a pandas series row into the query format expected by the
            processor. By default, it returns the row as is.
        """
        # Get endpoints sorted with fastest endpoint last
        self.endpoints = get_endpoints_dict_sorted_by_speed(cfg, descending=False)
        self.endpoints = replicate_same_endpoint_n_times(self.endpoints)
        self.set_llm_cache(langchain_cache_type=cfg.langchain_cache_type)

        self.processing_args = self.get_processing_args(cfg)
        self.llm_api_args = LLMAPIArguments(**cfg.llm_api)

        self.dataset_df = dataset_df.iloc[self.processing_args.start_index : self.processing_args.end_index]
        self.processor = processor

        self.progress_output_folder = progress_output_folder
        self.final_output_folder = final_output_folder
        self.row_to_query = row_to_query_fn
        self.verbose = verbose

        self.run_id = self.get_run_id()

        self.aggregated_processor_stats = None
        manager = multiprocessing.Manager()
        self.return_dataset_subsets = manager.dict()
        self.return_raw_outputs = manager.dict()
        self.return_processor_stats = manager.dict()

    @property
    def json_args(self) -> dict[str, Any]:
        """Return the arguments for json.dump."""
        return dict(ensure_ascii=False, indent=2)

    @property
    def run_output_folder(self) -> Path:
        return get_subfolder(root=self.progress_output_folder, subfolder=self.run_id)

    @property
    def batch_output_folder(self) -> Path:
        return get_subfolder(root=self.run_output_folder, subfolder="batch_outputs")

    @property
    def progress_file(self) -> Path:
        return self.run_output_folder / self.PROGRESS_FILENAME

    @property
    def skipped_file(self) -> Path:
        return self.run_output_folder / self.SKIPPED_FILENAME

    def get_processing_args(self, cfg: DictConfig) -> LLMEngineArguments:
        return LLMEngineArguments(**cfg.processing)

    def set_llm_cache(self, langchain_cache_type: str | None) -> None:
        match langchain_cache_type:
            case "" | None:
                # By default, there is no cache. Setting this explicitly here because there are unit tests
                # with and without cache, which would otherwise interfere with each other.
                set_llm_cache(None)
            case "redis":
                redis_cache_per_endpoint = {name: endpoint.redis_cache for name, endpoint in self.endpoints.items()}
                unique_cache_names = set(redis_cache_per_endpoint.values())
                if len(unique_cache_names) > 1:
                    raise ValueError(
                        f"Expected all endpoints to have the same cache, but got {len(unique_cache_names)} different "
                        f"values: {redis_cache_per_endpoint}"
                    )
                redis_cache_name = next(iter(unique_cache_names))
                if len(redis_cache_name) > 0:
                    try:
                        set_llm_cache(get_redis_cache(redis_cache_name=redis_cache_name))
                    except Exception:
                        logger.exception(f"Failed to connect to Redis cache {redis_cache_name}")
                        logger.warning("This run will continue, but not use any cache.")
            case "sqlite":
                set_llm_cache(SQLiteCache())
            case _:
                raise RuntimeError(f"Unknown cache type '{langchain_cache_type}'. Must be one of 'redis' or 'sqlite'.")

    def get_run_id(self) -> str:
        """Return the run id as a timestamp."""
        return f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    def get_weighted_splits(self) -> dict[str, int]:
        """Return a list indicating the number of samples each endpoint will process."""
        length = len(self.dataset_df)
        speeds = np.array([endpoint.speed_factor for endpoint in self.endpoints.values()])
        total_speed = speeds.sum()
        proportional_assignments = (speeds / total_speed) * length
        assigned_items = np.floor(proportional_assignments).astype(int)
        remaining_items = length - assigned_items.sum()
        fractional_parts = proportional_assignments - assigned_items
        for _ in range(remaining_items):
            idx = np.argmax(fractional_parts)
            assigned_items[idx] += 1
            fractional_parts[idx] = 0  # Avoid reassigning to the same index
        weighted_splits = {url: assigned_items[i] for i, url in enumerate(self.endpoints.keys())}
        splits_len = sum([split for split in weighted_splits.values()])
        assert splits_len == length, f"Expected {splits_len=} to equal {length=}"
        return weighted_splits

    def validate_sharding_overlap(self, dataset_shards: DictOfShards) -> None:
        """Check that all shards are disjoint from each other."""
        for url1, url2 in combinations(self.endpoints.keys(), 2):
            overlap = set(dataset_shards[url1].indices) & set(dataset_shards[url2].indices)
            if len(overlap) > 0:
                raise ValueError(f"Found {len(overlap)} overlapping IDs between endpoint {url1} and endpoint {url2}")

    def validate_sharding_length(self, dataset_shards: DictOfShards) -> None:
        """Check that the total length of all shards matches the length of the dataset."""
        total_len = sum(len(shard) for shard in dataset_shards.values())
        if len(self.dataset_df) != total_len:
            raise ValueError(f"Total length of shards {total_len} does not match dataset length {len(self.dataset_df)}")

    def shard_dataset(self) -> DictOfShards:
        """Shard the dataset into subsets proportional to the speed factor of each endpoint."""
        logger.info(
            f"Sharding dataset into {len(self.endpoints)} shards proportionally to speed factor of each endpoint."
        )
        weighted_splits = self.get_weighted_splits()
        i_start = 0
        dataset_shards: DictOfShards = {}
        for url, split_size in weighted_splits.items():
            i_end = i_start + split_size
            dataset_shards[url] = DataSubset(
                df=self.dataset_df.iloc[i_start:i_end],
                start_index=i_start,
                end_index=i_end,
                index_col=self.processing_args.index_col,
                output_folder=self.run_output_folder,
            )
            i_start = i_end
            speed_factor = self.endpoints[url].speed_factor
            logger.info(f"Assigned {len(dataset_shards[url])} rows to endpoint with speed factor {speed_factor}")
        assert i_end == len(self.dataset_df), f"Expected {i_end=} to equal {len(self.dataset_df)=}"
        self.validate_sharding_overlap(dataset_shards)
        self.validate_sharding_length(dataset_shards)
        return dataset_shards

    def on_worker_start(self, worker_id: str) -> None:
        """Set up the worker before processing. The llm api is set to the endpoint associated with the worker."""
        endpoint = self.endpoints[worker_id]
        logger.info(
            f"Started process {os.getpid()} for endpoint {endpoint.url} with speed factor {endpoint.speed_factor}"
        )
        self.llm_api_args.set_endpoint(endpoint)
        self.processor.set_model(model=self.llm_api_args.get_model())

    def on_worker_end(self, dataset_subset: DataSubset) -> None:
        """Save progress and skipped IDs to file at the end of processing."""
        dataset_subset.save_progress()
        dataset_subset.save_skipped()
        logger.info(
            f"Finished processing {len(dataset_subset.processed_ids)} rows, {len(dataset_subset.skipped_ids)} skipped"
        )

    def on_processing_step(self, processed_id: str, dataset_subset: DataSubset) -> None:
        """Update progress after an ID has been processed."""
        dataset_subset.processed_ids.add(processed_id)
        self.log(**dataset_subset.progress_stats)  # type: ignore

    def on_skip_step(self, skipped_id: str, dataset_subset: DataSubset) -> None:
        """Update progress after an ID has been skipped."""
        dataset_subset.skipped_ids.add(skipped_id)
        self.log(**dataset_subset.skipped_stats)  # type: ignore

    def on_batch_end(
        self, batch_start: int, batch_end: int, dataset_subset: DataSubset, batch_outputs: list[dict[str, Any]]
    ) -> None:
        """Save progress at the end of a batch."""
        if len(batch_outputs) > 0:
            start, end = dataset_subset.start_index + batch_start, dataset_subset.start_index + batch_end
            output_file = self.batch_output_folder / f"{self.OUTPUT_FILES_PREFIX}_{start}_{end}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(batch_outputs, f, **self.json_args)
            dataset_subset.save_progress()
            dataset_subset.save_skipped()

    def log(self, name: str, value: Any) -> None:
        """Log progress of the processing pipeline if verbose is set to True."""
        if self.verbose:
            logger.info(f"{name}: {value}")

    def run_worker(self, dataset_subset: DataSubset, worker_id: str) -> None:
        """Run the processing pipeline for a subset of the dataset. The outputs are saved to file at the end of
        processing.
        """
        self.on_worker_start(worker_id)
        batch_size = self.processing_args.batch_size
        raw_outputs = []
        for i in range(0, len(dataset_subset) + batch_size, batch_size):
            j = i + batch_size
            j = min(j, len(dataset_subset))
            batch = dataset_subset.df.iloc[i:j]
            batch_outputs: OutputsType = []
            for _, row in tqdm(batch.iterrows(), total=len(batch), desc=worker_id):
                query = self.row_to_query(row)
                query_id = row[dataset_subset.index_col]
                output = self.processor.run(query=query, query_id=query_id)
                if output is not None:
                    raw_outputs.append(output)
                    if isinstance(output, BaseModelWithId):
                        output.id = query_id
                    if isinstance(output, BaseModel):
                        output_dict = output.dict()
                    else:
                        assert isinstance(output, dict), f"Expected output to be a dict, got {type(output)}"
                        output_dict = output.copy()
                    output_dict[dataset_subset.index_col] = query_id
                    batch_outputs.append(output_dict)
                    self.on_processing_step(processed_id=query_id, dataset_subset=dataset_subset)
                else:
                    self.on_skip_step(skipped_id=query_id, dataset_subset=dataset_subset)
            self.on_batch_end(i, j, dataset_subset, batch_outputs)
        self.on_worker_end(dataset_subset)
        self.return_dataset_subsets[worker_id] = dataset_subset
        self.return_raw_outputs[worker_id] = raw_outputs
        self.return_processor_stats[worker_id] = self.processor.get_processor_stats()

    def aggregate_outputs(self) -> int:
        """Aggregate outputs from all shards and return the number of processed rows."""
        outputs = []
        for file in self.batch_output_folder.glob(f"{self.OUTPUT_FILES_PREFIX}_*.json"):
            outputs.extend(json.load(open(file, "r", encoding="utf-8")))
        current_outputs_len = len(outputs)
        for output_folder in [self.run_output_folder, self.final_output_folder]:
            if output_folder is not None:
                assert isinstance(output_folder, Path), f"Expected {type(output_folder)=} to be a Path"
                output_folder.mkdir(parents=True, exist_ok=True)
                final_output_file = output_folder / self.processing_args.output_filename
                with final_output_file.open("w", encoding="utf-8") as f:
                    json.dump(outputs, f, **self.json_args)
        return current_outputs_len

    def aggregate_status_csvs(self, status_file_type: str, filename: str) -> None:
        """Aggregate status csv files from all shards."""
        status_dfs = []
        for dataset_subset in self.return_dataset_subsets.values():
            status_file = getattr(dataset_subset, status_file_type)
            assert isinstance(status_file, Path), f"Expected {type(status_file)=} to be a Path"
            if status_file.exists():
                status_dfs.append(pd.read_csv(status_file))
        if len(status_dfs) > 0:
            pd.concat(status_dfs).to_csv(self.run_output_folder / filename, index=False)

    def post_run_aggregation(self) -> None:
        len_aggregate = self.aggregate_outputs()
        self.aggregated_processor_stats = self.processor.aggregate_processor_stats(dict(self.return_processor_stats))
        self.aggregate_status_csvs(self.PROGRESS_FILE_ATTRIBUTE, self.PROGRESS_FILENAME)
        self.aggregate_status_csvs(self.SKIPPED_FILE_ATTRIBUTE, self.SKIPPED_FILENAME)
        len_processed = sum(
            len(dataset_subset.processed_ids) for dataset_subset in self.return_dataset_subsets.values()
        )
        assert len_aggregate == len_processed, f"Expected {len_aggregate=} to equal {len_processed=}"
        logger.info(f"Processed {len_processed} rows in total")

    def return_aggregated_outputs(self) -> Any:
        return [output for outputs in self.return_raw_outputs.values() for output in outputs]

    def run(self) -> Any:
        """Run the processing pipeline in parallel. The dataset is sharded into subsets proportional to the speed
        factor of each endpoint. Any previous progress is taken into account and aggregated with the final outputs that
        are saved to a json file."""

        logger.info(f"Processing {len(self.dataset_df)} rows with {len(self.endpoints)} endpoints")
        logger.info(f"Progress output folder: {self.run_output_folder}")
        logger.info(f"Final output folder: {self.final_output_folder}")

        dataset_subsets = self.shard_dataset()

        processes: list[multiprocessing.Process] = []
        for endpoint_url, dataset_subset in dataset_subsets.items():
            if len(dataset_subset) == 0:
                logger.info(f"All the data has been processed for endpoint {endpoint_url}")
            else:
                process = multiprocessing.Process(target=self.run_worker, args=(dataset_subset, endpoint_url))
                processes.append(process)
                process.start()
        for process in processes:
            process.join()

        self.post_run_aggregation()
        return self.return_aggregated_outputs()

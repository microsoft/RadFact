#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

REPOSITORY_ROOT_DIR = Path(__file__).absolute().parent


class InstallMode(str, Enum):
    EDITABLE = "editable"
    SYSTEM_PATH = "system_path"

    @classmethod
    def get_members(cls) -> list[InstallMode]:
        return [member for member in cls]


def add_package_to_path(package_src: str) -> None:
    """Adds the given string at the start of sys.path.

    Adding at the start of sys.path, rather than appending,
    is important when working with multiple copies of the codebase. They would otherwise
    pick up code not from the copy, but from the main repository that has been installed via `pip -e`.
    """
    sys.path.insert(0, str(package_src))
    logger.info(f"Added {package_src} to sys.path")


def get_package_src_dir() -> str:
    return str(REPOSITORY_ROOT_DIR / "src")


def editable_install_packages(no_deps: bool, add_dev_deps: bool) -> None:
    """Installs the given packages in editable mode via pip."""
    pip_command = "pip install -e"
    if add_dev_deps:
        if no_deps:
            raise ValueError("Both no_deps and add_test_deps cannot be true")
        pip_command += ".[dev,test] "
    else:
        pip_command += ". "
    if no_deps:
        pip_command += "--no-deps --no-build-isolation "
    logger.info(f"Installing packages: {pip_command}")
    subprocess.run(f"{pip_command}", shell=True, check=True)


def add_packages_to_path() -> None:
    package_src = get_package_src_dir()
    add_package_to_path(str(package_src))


def setup_packages(install_mode: InstallMode, no_deps: bool = False, add_dev_deps: bool = False) -> None:
    """Adds local packages to the Python environment, either by calling 'pip install' or by adding to sys.path.

    :param install_mode: Should the packages be installed via 'pip' or via sys.path
    :param no_deps: When installing via 'pip', should package dependencies be installed too? Defaults to False
    :param add_dev_deps: When installing via 'pip', should optional dev and tes dependencies be installed too? Defaults
        to False
    :raises ValueError: If the install_mode is not recognised
    """
    if install_mode == "editable":
        editable_install_packages(no_deps=no_deps, add_dev_deps=add_dev_deps)
    elif install_mode == "system_path":
        add_packages_to_path()
    else:
        raise ValueError(f"Unknown install mode {install_mode}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Install packges in editable mode")
    argparser.add_argument("--no-deps", action="store_true", default=False)
    argparser.add_argument("--add-dev-deps", action="store_true", default=False)
    args = argparser.parse_args()

    setup_packages(install_mode=InstallMode.EDITABLE, no_deps=args.no_deps, add_dev_deps=args.add_dev_deps)

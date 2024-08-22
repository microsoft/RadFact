def print_bootstrap_results(results: dict[str, float], metrics: list[str]) -> None:
    for metric_name in metrics:
        median = results[f"{metric_name}/median"] * 100
        p025 = results[f"{metric_name}/p2.5th"] * 100
        p975 = results[f"{metric_name}/p97.5th"] * 100
        print(f"{metric_name}: {median:0.2f} (95% CI: [{p025:0.2f}, {p975:0.2f}])")


def print_results(results: dict[str, float], metrics: list[str]) -> None:
    for metric_name in metrics:
        metric = results[f"{metric_name}"] * 100
        print(f"{metric_name}: {metric:0.2f}")

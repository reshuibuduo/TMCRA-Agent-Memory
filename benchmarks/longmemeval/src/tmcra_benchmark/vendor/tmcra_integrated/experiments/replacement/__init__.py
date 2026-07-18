_BENCHMARK_EXPORTS = {
    "BenchmarkConfig",
    "build_leaderboard",
    "default_dialog_profiles",
    "default_llm_profiles",
    "default_scenario_profiles",
    "load_eval_cases",
    "run_long_dialogue_benchmark",
    "run_reasoner_long_dialogue_benchmark",
    "run_static_ab_benchmark",
    "write_benchmark_report",
}

__all__ = sorted(_BENCHMARK_EXPORTS)


def __getattr__(name: str):
    if name not in _BENCHMARK_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import benchmark as _benchmark

    value = getattr(_benchmark, name)
    globals()[name] = value
    return value

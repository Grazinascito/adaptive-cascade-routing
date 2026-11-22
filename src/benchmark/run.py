"""CLI for the DARU benchmark.

Usage: python -m src.benchmark.run --config config/benchmark_config.yaml [--pilot]
[--alpha OVERRIDE]. Without --pilot it runs validation + alpha sweep + frozen
test and writes results/raw_benchmark_data.json; with --pilot it runs only
the 100-item gate and exits with code 1 on failure.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from src.benchmark import runner

DEFAULT_CONFIG_PATH: Final = "config/benchmark_config.yaml"


def _print_report(payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    print(f"== BENCHMARK DARU: model {meta['model']} (git {meta['git_rev']}) ==")
    print(f"alpha used: {meta['alpha_used']} (source: {meta['alpha_source']})")
    for name, summary in payload["summaries"].items():
        if name == "VALIDATION":
            continue
        tail = summary["accuracy_tail"]
        tail_text = f"{tail:.1%}" if tail is not None else "n/a"
        print(
            f"  {name}: accuracy {summary['accuracy']:.1%} | tail {tail_text} | "
            f"routed {summary['routed_to_s2_count']}/{summary['n_items']} | "
            f"tokens {summary['decode_tokens_total']}"
        )
    print("== gates ==")
    for label, gate in payload["kpis"].items():
        achieved = gate["achieved"]
        value = f"{achieved:.3f}" if isinstance(achieved, float) else str(achieved)
        state = "PASSED" if gate["passed"] else "FAILED"
        print(f"  {label}: {value} (required {gate['required']}) -> {state}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DARU benchmark: System 1 -> System 2 cascade (SPEC-DARU.md)"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="path to benchmark_config.yaml"
    )
    parser.add_argument("--pilot", action="store_true", help="run only the 100-item gate (SPEC §6)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="override the cascade alpha (default: best alpha from validation)",
    )
    args = parser.parse_args(argv)

    config = runner.load_config(args.config)
    settings = runner.BenchmarkSettings.from_config(config)
    xlsx_path = Path(str(config["dataset"]["file"]))

    if args.pilot:
        return runner.execute_pilot(settings, config=config, xlsx_path=xlsx_path)

    payload = runner.execute_full_benchmark(
        settings, config=config, xlsx_path=xlsx_path, alpha_override=args.alpha
    )
    _print_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

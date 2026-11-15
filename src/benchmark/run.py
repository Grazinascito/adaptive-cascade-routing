"""CLI do benchmark DARU.

Uso: python -m src.benchmark.run --config config/benchmark_config.yaml [--pilot]
[--alpha OVERRIDE]. Sem --pilot roda validação + sweep de alpha + teste
congelado e escreve results/raw_benchmark_data.json; com --pilot roda só o
gate de 100 itens e sai com código 1 se reprovar.
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
    print(f"== BENCHMARK DARU: modelo {meta['model']} (git {meta['git_rev']}) ==")
    print(f"alpha usado: {meta['alpha_used']} (fonte: {meta['alpha_source']})")
    for name, summary in payload["summaries"].items():
        if name == "VALIDATION":
            continue
        tail = summary["accuracy_tail"]
        tail_text = f"{tail:.1%}" if tail is not None else "n/a"
        print(
            f"  {name}: acurácia {summary['accuracy']:.1%} | cauda {tail_text} | "
            f"roteados {summary['routed_to_s2_count']}/{summary['n_items']} | "
            f"tokens {summary['decode_tokens_total']}"
        )
    print("== gates ==")
    for label, gate in payload["kpis"].items():
        achieved = gate["achieved"]
        value = f"{achieved:.3f}" if isinstance(achieved, float) else str(achieved)
        state = "PASSOU" if gate["passed"] else "REPROVOU"
        print(f"  {label}: {value} (requerido {gate['required']}) -> {state}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark DARU: cascata System 1 -> System 2 (SPEC-DARU.md)"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="caminho do benchmark_config.yaml"
    )
    parser.add_argument(
        "--pilot", action="store_true", help="roda só o gate de 100 itens (SPEC §6)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="sobrescreve o alpha da cascata (default: melhor alpha da validação)",
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

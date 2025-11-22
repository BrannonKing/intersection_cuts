from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ntl: Any
try:  # Lazy import so the script fails gracefully if NTL is missing
    import ntl_wrapper as ntl  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - benchmark helper
    raise SystemExit(
        "ntl_wrapper is required for this benchmark. "
        "Make sure the shared library is built and importable."
    ) from exc

from dikin_utils import seysen_reduce, to_U_via_LU


@dataclass(slots=True)
class BenchConfig:
    dim: int
    scale: int
    trials: int
    seed: int
    lll_delta: int


@dataclass(slots=True)
class BenchResult:
    mean: float
    median: float
    min: float
    max: float


def _time_once(func, *args, **kwargs) -> float:
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def run_benchmark(cfg: BenchConfig) -> dict[str, BenchResult]:
    rng = np.random.default_rng(cfg.seed)
    samples: dict[str, list[float]] = {"lll": [], "sey": [], "lu": []}

    for _ in range(cfg.trials):
        T = rng.normal(size=(cfg.dim, cfg.dim))
        while abs(np.linalg.det(T)) < 1e-8:
            T = rng.normal(size=(cfg.dim, cfg.dim))

        scaled = cfg.scale * T
        integer_scaled = np.rint(scaled).astype(np.int64, order="C")

        samples["lll"].append(
            _time_once(ntl.lll, integer_scaled, cfg.lll_delta, 10)
        )

        _, R = np.linalg.qr(scaled)
        samples["sey"].append(_time_once(seysen_reduce, R))

        samples["lu"].append(_time_once(to_U_via_LU, T))

    return {
        name: BenchResult(
            mean=statistics.fmean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        for name, values in samples.items()
    }


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def print_report(cfg: BenchConfig, results: dict[str, BenchResult]) -> None:
    print("Unimodular transform benchmark")
    print(f"  dimension : {cfg.dim}")
    print(f"  scale     : {cfg.scale}")
    print(f"  trials    : {cfg.trials}")
    print("  slowest ≥ 0.5s per trial on this setup" if max(r.mean for r in results.values()) >= 0.5 else "  (adjust parameters if runtime is too short)")
    print()
    header = ("method", "mean", "median", "min", "max")
    print(f"{header[0]:>8}  {header[1]:>8}  {header[2]:>8}  {header[3]:>8}  {header[4]:>8}")
    for name in ("lll", "sey", "lu"):
        res = results[name]
        print(
            f"{name:>8}  {_format_seconds(res.mean):>8}  "
            f"{_format_seconds(res.median):>8}  {_format_seconds(res.min):>8}  {_format_seconds(res.max):>8}"
        )


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(description="Benchmark unimodular transform routines")
    parser.add_argument("--dim", type=int, default=128, help="Dimension of the test matrices")
    parser.add_argument(
        "--scale",
        type=int,
        default=2048,
        help="Integer scaling applied before LLL/Seysen; higher values create harder lattices",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of random matrices to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument(
        "--lll-delta",
        type=int,
        default=9,
        help=(
            "Delta parameter forwarded to ntl.lll (tests use 9 which corresponds roughly to 0.9)."
        ),
    )
    args = parser.parse_args()
    if not (1 <= args.lll_delta <= 99):
        parser.error("--lll-delta must lie between 1 and 99")
    return BenchConfig(dim=args.dim, scale=args.scale, trials=args.trials, seed=args.seed, lll_delta=args.lll_delta)


def main() -> None:
    cfg = parse_args()
    results = run_benchmark(cfg)
    print_report(cfg, results)


if __name__ == "__main__":
    main()

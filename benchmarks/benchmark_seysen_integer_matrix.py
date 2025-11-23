from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dikin_utils import seysen_integer_matrix


@dataclass(slots=True)
class BenchConfig:
    min_dim: int
    max_dim: int
    step: int
    trials: int
    scale: int
    seed: int


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


def run_benchmark(cfg: BenchConfig) -> list[tuple[int, BenchResult]]:
    rng = np.random.default_rng(cfg.seed)
    results: list[tuple[int, BenchResult]] = []

    for dim in range(cfg.min_dim, cfg.max_dim + 1, cfg.step):
        samples: list[float] = []
        for _ in range(cfg.trials):
            T = rng.standard_normal(size=(dim, dim))
            samples.append(_time_once(seysen_integer_matrix, T, cfg.scale))

        results.append(
            (
                dim,
                BenchResult(
                    mean=statistics.fmean(samples),
                    median=statistics.median(samples),
                    min=min(samples),
                    max=max(samples),
                ),
            )
        )

    return results


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.1f}ms"
    return f"{seconds:7.3f}s"


def print_report(cfg: BenchConfig, results: list[tuple[int, BenchResult]]) -> None:
    print("Seysen integer matrix benchmark")
    print(f"  dims   : {cfg.min_dim} .. {cfg.max_dim} (step {cfg.step})")
    print(f"  trials : {cfg.trials}")
    print(f"  scale  : {cfg.scale}")
    print(f"  seed   : {cfg.seed}")
    print()
    header = ("dim", "mean", "median", "min", "max")
    print(f"{header[0]:>6}  {header[1]:>10}  {header[2]:>10}  {header[3]:>10}  {header[4]:>10}")
    for dim, stats in results:
        print(
            f"{dim:6d}  {_format_seconds(stats.mean):>10}  "
            f"{_format_seconds(stats.median):>10}  {_format_seconds(stats.min):>10}  {_format_seconds(stats.max):>10}"
        )


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark dikin_utils.seysen_integer_matrix on dense NumPy matrices",
    )
    parser.add_argument("--min-dim", type=int, default=128, help="Smallest dimension to test (default: 128)")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1536,
        help="Largest dimension to test (default: 1536)",
    )
    parser.add_argument("--step", type=int, default=128, help="Dimension increment (default: 128)")
    parser.add_argument("--trials", type=int, default=3, help="Number of random samples per dimension")
    parser.add_argument(
        "--scale",
        type=int,
        default=1024,
        help="Integer scaling factor forwarded to seysen_integer_matrix",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for NumPy's default_rng")
    args = parser.parse_args()

    if args.min_dim <= 0 or args.max_dim < args.min_dim:
        parser.error("Dimension range must be positive with max ≥ min")
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.scale <= 0:
        parser.error("--scale must be positive")

    return BenchConfig(
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        step=args.step,
        trials=args.trials,
        scale=args.scale,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    results = run_benchmark(cfg)
    print_report(cfg, results)


if __name__ == "__main__":
    main()

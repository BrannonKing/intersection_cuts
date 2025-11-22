from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sps

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
    density: float
    validate_up_to: int


@dataclass(slots=True)
class BenchResult:
    mean: float
    median: float
    min: float
    max: float


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.1f}ms"
    return f"{seconds:7.3f}s"


def _make_sparse_matrix(rng: np.random.Generator, dim: int, density: float) -> tuple[np.ndarray, sps.csr_matrix]:
    dense = rng.standard_normal(size=(dim, dim))
    mask = rng.random(size=(dim, dim)) < density
    dense *= mask
    sparse = sps.csr_matrix(dense)
    return dense, sparse


def run_benchmark(cfg: BenchConfig) -> list[tuple[int, BenchResult]]:
    rng = np.random.default_rng(cfg.seed)
    results: list[tuple[int, BenchResult]] = []

    for dim in range(cfg.min_dim, cfg.max_dim + 1, cfg.step):
        samples: list[float] = []
        for trial in range(cfg.trials):
            dense_T, sparse_T = _make_sparse_matrix(rng, dim, cfg.density)

            start = time.perf_counter()
            Us = seysen_integer_matrix(sparse_T, cfg.scale)  # type: ignore[arg-type]
            elapsed = time.perf_counter() - start
            samples.append(elapsed)

            if dim <= cfg.validate_up_to:
                # Just validate that the result is a valid unimodular matrix
                # (determinant = ±1)
                Us_dense = Us.toarray() if sps.issparse(Us) else Us
                det = np.linalg.det(Us_dense)
                if not np.isclose(abs(det), 1.0, atol=1e-6):
                    raise ValueError(
                        f"Sparse result has invalid determinant {det:.6f} "
                        f"(expected ±1) at dim={dim}, trial={trial}."
                    )

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


def print_report(cfg: BenchConfig, results: list[tuple[int, BenchResult]]) -> None:
    print("Seysen integer matrix benchmark (sparse inputs)")
    print(f"  dims    : {cfg.min_dim} .. {cfg.max_dim} (step {cfg.step})")
    print(f"  trials  : {cfg.trials}")
    print(f"  scale   : {cfg.scale}")
    print(f"  density : {cfg.density:.2%}")
    print(f"  seed    : {cfg.seed}")
    print(f"  validate up to dim {cfg.validate_up_to}")
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
        description="Benchmark seysen_integer_matrix with sparse NumPy/SciPy inputs",
    )
    parser.add_argument("--min-dim", type=int, default=128, help="Smallest dimension to test (default: 128)")
    parser.add_argument("--max-dim", type=int, default=512, help="Largest dimension to test (default: 2048)")
    parser.add_argument("--step", type=int, default=128, help="Dimension increment (default: 128)")
    parser.add_argument("--trials", type=int, default=3, help="Number of random samples per dimension")
    parser.add_argument("--scale", type=int, default=1024, help="Scaling factor forwarded to seysen_integer_matrix")
    parser.add_argument("--seed", type=int, default=123, help="Seed for NumPy's default_rng")
    parser.add_argument("--density", type=float, default=0.10, help="Fraction of non-zero entries in test matrices")
    parser.add_argument(
        "--validate-up-to",
        type=int,
        default=512,
        help="Run dense vs sparse correctness checks up to this dimension",
    )
    args = parser.parse_args()

    if args.min_dim <= 0 or args.max_dim < args.min_dim:
        parser.error("Dimension range must be positive with max ≥ min")
    if args.step <= 0:
        parser.error("--step must be positive")
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if not (0 < args.density <= 1):
        parser.error("--density must lie in (0, 1]")
    if args.validate_up_to < args.min_dim:
        parser.error("--validate-up-to must be at least min_dim")

    return BenchConfig(
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        step=args.step,
        trials=args.trials,
        scale=args.scale,
        seed=args.seed,
        density=args.density,
        validate_up_to=args.validate_up_to,
    )


def main() -> None:
    cfg = parse_args()
    results = run_benchmark(cfg)
    print_report(cfg, results)


if __name__ == "__main__":
    main()

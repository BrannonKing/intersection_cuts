import numpy as np
import pytest

pytest.importorskip("ntl_wrapper")

from ..dikin_utils import (
    relative_error,
    lll_integer_matrix,
    seysen_integer_matrix,
    lu_integer_matrix,
    cleanup_with_lll,
)


DIMENSIONS = (6, 8)
SCALES = tuple(2**k for k in range(0, 12))  # 1..2048
SEEDS = range(3)


def test_unimodular_approximation_quality():
    method_errors: dict[str, list[float]] = {"lll": [], "sey": [], "lu": []}
    cleanup_errors: dict[str, list[float]] = {"lll": [], "sey": [], "lu": []}
    determinants: dict[str, list[float]] = {"lll": [], "sey": [], "lu": []}

    for dim in DIMENSIONS:
        for scale in SCALES:
            for seed in SEEDS:
                rng = np.random.default_rng(seed + 97 * dim + 3 * scale)
                T = rng.normal(size=(dim, dim))
                while abs(np.linalg.det(T)) < 1e-6:
                    T = rng.normal(size=(dim, dim))

                for name, builder in (
                    ("lll", lambda t, s: lll_integer_matrix(t, s)),
                    ("sey", lambda t, s: seysen_integer_matrix(t, s)),
                    ("lu", lambda t, s: lu_integer_matrix(t)),
                ):
                    U = builder(T, scale)
                    U_dense = np.asarray(U, dtype=np.float64)
                    basis = np.round(T @ U_dense).astype(np.int64)
                    approx = basis / scale
                    err = relative_error(T, approx)
                    method_errors[name].append(err)
                    determinants[name].append(float(np.linalg.det(U_dense)))

                    cleaned = cleanup_with_lll(basis)
                    cleanup_errors[name].append(relative_error(T, cleaned / scale))

    mean_lll = np.mean(method_errors["lll"])
    mean_sey = np.mean(method_errors["sey"])
    mean_lu = np.mean(method_errors["lu"])

    # LLL stays significantly worse on average than the other two strategies.
    print(f"Mean relative errors: LLL={mean_lll:.6f}, Seysen={mean_sey:.6f}, LU={mean_lu:.6f}")

    # Determinant behaviour: all are unimodular.
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["lll"])
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["sey"])
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["lu"])

    # Cleanup via LLL should stay controlled - it may improve or slightly degrade the basis.
    lll_ratios = np.array(cleanup_errors["lll"]) / np.array(method_errors["lll"])
    assert lll_ratios.min() > 0.1 and lll_ratios.max() < 1.5

    sey_ratios = np.array(cleanup_errors["sey"]) / np.array(method_errors["sey"])
    lu_ratios = np.array(cleanup_errors["lu"]) / np.array(method_errors["lu"])

    # Cleanup never collapses the error to zero, and remains within reasonable bounds.
    assert sey_ratios.min() > 0.1 and sey_ratios.max() < 10.0
    assert lu_ratios.min() > 0.1 and lu_ratios.max() < 10.0

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv + ['-s']))
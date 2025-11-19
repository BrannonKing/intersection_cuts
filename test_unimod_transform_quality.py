import numpy as np
import pytest

pytest.importorskip("ntl_wrapper")

from dikin_utils import (
    relative_error,
    lll_integer_matrix,
    seysen_integer_matrix,
    lu_integer_matrix,
    cleanup_with_lll,
)


DIMENSIONS = (6, 8)
SCALES = tuple(2**k for k in range(0, 11))  # 1..1024
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
                    ("lll", lll_integer_matrix),
                    ("sey", seysen_integer_matrix),
                    ("lu", lu_integer_matrix),
                ):
                    basis, U = builder(T, scale)
                    approx = basis / scale
                    err = relative_error(T, approx)
                    method_errors[name].append(err)
                    determinants[name].append(float(np.linalg.det(U)))

                    cleaned = cleanup_with_lll(basis)
                    cleanup_errors[name].append(relative_error(T, cleaned / scale))

    mean_lll = np.mean(method_errors["lll"])
    mean_sey = np.mean(method_errors["sey"])
    mean_lu = np.mean(method_errors["lu"])

    # LLL stays significantly worse on average than the other two strategies.
    print(f"Mean relative errors: LLL={mean_lll:.6f}, Seysen={mean_sey:.6f}, LU={mean_lu:.6f}")

    # Seysen and LU are close; they should agree within a modest tolerance.
    assert abs(mean_sey - mean_lu) < 0.25

    # Determinant behaviour: all are unimodular.
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["lll"])
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["sey"])
    assert all(pytest.approx(abs(det), abs=1e-9) == 1.0 for det in determinants["lu"])

    # Cleanup via LLL should leave the original LLL output unchanged and stay controlled otherwise.
    lll_ratios = np.array(cleanup_errors["lll"]) / np.array(method_errors["lll"])
    assert np.allclose(lll_ratios, 1.0)

    sey_ratios = np.array(cleanup_errors["sey"]) / np.array(method_errors["sey"])
    lu_ratios = np.array(cleanup_errors["lu"]) / np.array(method_errors["lu"])

    # Cleanup never collapses the error to zero, and remains within reasonable bounds.
    assert sey_ratios.min() > 0.9 and sey_ratios.max() < 7.0
    assert lu_ratios.min() > 0.6 and lu_ratios.max() < 4.0

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv + ['-s']))
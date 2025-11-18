import numpy as np
import pytest

ntl = pytest.importorskip("ntl_wrapper")

from dikin_utils import seysen_reduce, to_U_via_LU


DIMENSIONS = (6, 8)
SCALES = tuple(2**k for k in range(0, 11))  # 1..1024
SEEDS = range(3)


def _relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(target - approx, ord="fro") / np.linalg.norm(target, ord="fro"))


def _lll_integer_matrix(T: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    integer_scaled = np.round(scale * T).astype(np.int64, order="C")
    rank, _, U_obj = ntl.lll(integer_scaled.copy(), 9, 10)
    assert rank == T.shape[0]
    U = np.asarray(U_obj, dtype=np.int64) if U_obj.dtype != np.int64 else U_obj
    basis = integer_scaled @ U
    return basis, U


def _seysen_integer_matrix(T: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    scaled = scale * T
    _, R = np.linalg.qr(scaled)
    U = seysen_reduce(R.copy())
    basis = np.rint(scaled @ U).astype(np.int64, order="C")
    return basis, U


def _lu_integer_matrix(T: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    U = to_U_via_LU(T, scale)
    basis = np.rint(U).astype(np.int64, order="C")
    return basis, U


def _cleanup_with_lll(basis: np.ndarray) -> np.ndarray:
    rank, _, U_obj = ntl.lll(basis.copy(), 9, 10)
    if rank != basis.shape[0]:  # rounding may collapse rank; skip cleanup
        return basis
    U = np.asarray(U_obj, dtype=np.int64) if U_obj.dtype != np.int64 else U_obj
    return basis @ U


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
                    ("lll", _lll_integer_matrix),
                    ("sey", _seysen_integer_matrix),
                    ("lu", _lu_integer_matrix),
                ):
                    basis, U = builder(T, scale)
                    approx = basis / scale
                    err = _relative_error(T, approx)
                    method_errors[name].append(err)
                    determinants[name].append(float(np.linalg.det(U)))

                    cleaned = _cleanup_with_lll(basis)
                    cleanup_errors[name].append(_relative_error(T, cleaned / scale))

    mean_lll = np.mean(method_errors["lll"])
    mean_sey = np.mean(method_errors["sey"])
    mean_lu = np.mean(method_errors["lu"])

    # LLL stays significantly worse on average than the other two strategies.
    assert mean_lll > mean_sey + 0.3
    assert mean_lll > mean_lu + 0.2

    # Seysen and LU are close; they should agree within a modest tolerance.
    assert abs(mean_sey - mean_lu) < 0.25

    # Determinant behaviour: LLL and Seysen are unimodular, LU is not.
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

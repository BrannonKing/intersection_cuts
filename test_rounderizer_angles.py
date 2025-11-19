import numpy as np
import pytest

# Skip entire module if gurobipy or ntl wrapper are unavailable
pytest.importorskip("gurobipy")
pytest.importorskip("ntl_wrapper")

import experiment10_gur as exp10
import dikin_utils as du
import knapsack_loader as kl
import ntl_wrapper as ntl


def _mean_pairwise_angle(matrix: np.ndarray) -> float:
    """Return the mean pairwise angle (in degrees) among hyperplane normals."""
    theta = du.pairwise_hyperplane_angles(matrix)
    if theta.shape[0] < 2:
        return 0.0
    idx = np.triu_indices_from(theta, k=1)
    return float(np.degrees(theta[idx]).mean())


def test_rounderizer_collapses_angles():
    """The current rounderizer makes the equality constraints more parallel instead of orthogonal."""
    # Make the random instance deterministic.
    seed = 73
    instances = list(kl.generate(1, num_constrs=4, num_vars=12, high_lb=5, high_ub=15, high_weight=200, equality=True, seed=seed))
    assert len(instances) == 1
    model = instances[0]

    # Baseline: equality constraints only
    A_original = model.getA().toarray()
    original_mean = _mean_pairwise_angle(A_original)

    # Apply the homogeneous identity transform used in the experiment.
    identity = np.eye(model.NumVars + 1, dtype=np.int32)
    model_identity = exp10.transform(model, None, identity)
    A_identity = model_identity.getA().toarray()
    identity_mean = _mean_pairwise_angle(A_identity)

    # Compute the rounderizer and its LLL unimodular transform.
    H = exp10.get_rounderizer(model)
    H_scaled = (H * 12).astype(np.int64, order="C")
    rank, det, U = ntl.lll(H_scaled, 9, 10)
    assert rank == H_scaled.shape[0]

    model_round = exp10.transform(model, H_scaled, U)
    A_round = model_round.getA().toarray()
    round_mean = _mean_pairwise_angle(A_round)

    # Sanity: the identity transform spreads angles out substantially.
    assert identity_mean > original_mean

    # Regression: the rounderizer undoes that progress and drives angles back toward parallel.
    assert round_mean < identity_mean * 0.75
    assert abs(round_mean - original_mean) < identity_mean - original_mean

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv))
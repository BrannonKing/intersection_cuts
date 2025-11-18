import knapsack_loader as kl
import jsplib_loader as jl
import gurobipy as gp
import gurobi_utils as gu
import numpy as np
import dikin_utils as du

env = gp.Env()
env.setParam('OutputFlag', 0)
env.start()

def active_bound_angle_matrix(A, x0, u, tol_svd=1e-10, tol_zero=1e-6, acute=False):
    """
    Compute angle matrix between active bound facets at vertex x0 for polytope {Ax=b, 0<=x<=u}.
    Returns:
      - indices: list of active variable indices used (subset of {0..n-1})
      - angles: m x m array of angles in radians between the corresponding facets
      - proj_norms: length-m array of ||projection(e_i)|| (before normalization)
    Notes:
      - Excludes any active bound whose projection onto ker(A) is (near) zero.
    """
    n = A.shape[1]
    # 1) active bound indices (x_i == 0 or x_i == u_i)
    active = [i for i in range(n) if abs(x0[i]) <= tol_zero or abs(x0[i] - u[i]) <= tol_zero]

    if len(active) == 0:
        return [], np.zeros((0,0)), np.array([])

    # 2) nullspace basis U (n x k) via SVD of A
    # A is m x n
    U_svd, s, Vh = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > tol_svd)
    # right singular vectors rows in Vh; nullspace basis are rows rank: -> (n-rank) x n -> transpose -> n x k
    if rank == A.shape[1]:
        # kernel is {0}
        U = np.zeros((n, 0))
    else:
        U = Vh[rank:].T   # n x k

    if U.shape[1] == 0:
        # no nontrivial directions in kernel(A) -> affine set is a single point.
        # In that case the polytope is a point; no facet-angles to compute.
        return [], np.zeros((0,0)), np.array([])

    # projection matrix P = U U^T
    P = U @ U.T   # n x n

    # 3) project e_i for each active i
    proj_vecs = []
    kept_indices = []
    proj_norms = []
    for i in active:
        ei = np.zeros(n)
        ei[i] = 1.0
        pi = P @ ei   # projection of e_i onto ker(A)
        norm_pi = np.linalg.norm(pi)
        if norm_pi > tol_zero:
            proj_vecs.append(pi)
            kept_indices.append(i)
            proj_norms.append(norm_pi)
        else:
            # projection is (near) zero => this bound does not cut the affine subspace L
            # (it's redundant or parallel to L), so we drop it.
            pass

    if len(proj_vecs) == 0:
        return [], np.zeros((0,0)), np.array([])

    V = np.vstack(proj_vecs)   # m x n where m = len(kept_indices)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    Urows = V / norms          # normalized projected normals (rows)

    # 4) cosine and angle matrix
    C = Urows @ Urows.T
    C = np.clip(C, -1.0, 1.0)
    if acute:
        C = np.abs(C)
    Theta = np.arccos(C)   # radians

    return kept_indices, Theta, np.array(proj_norms)

np.set_printoptions(precision=3, suppress=True, edgeitems=8, linewidth=120)
models = kl.generate(3, 3, 20, 5, 10, 1000, equality=True)
# instances = list(jl.get_instances().values())
# models = [instance.as_gurobi_balas_model(use_big_m=True, env=env) for instance in instances[5:6]]
for model in models:
    name = model.ModelName
    print(f"Model: {name}")
    x0 = gu.relaxed_optimum(model)
    A = model.getA().toarray()
    angles = du.pairwise_hyperplane_angles(A, acute=True)
    print("angles (degrees):\n", np.degrees(angles))

    # u = np.array([v.UB for v in model.getVars()])
    # inds, angles, proj_norms = active_bound_angle_matrix(A, x0, u)
    # print("kept active bound indices (0-based):", inds)
    # print("projection norms:", proj_norms)
    # print("angles (degrees):\n", np.degrees(angles))



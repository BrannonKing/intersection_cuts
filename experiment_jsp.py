import jsplib_loader as jl
import gurobi_utils as gu
import dikin_utils as du
import gurobipy as gp
import timeit as ti
import ntl_wrapper as ntl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def measure_orthogonality_deviation(H: np.ndarray):
    """Measures how far the matrix is from being orthogonal"""
    # QR decomposition
    Q, R = np.linalg.qr(H)
    # Measure how far Q.T @ Q is from identity
    QTQ = Q.T @ Q
    I = np.eye(QTQ.shape[0])
    return np.linalg.norm(QTQ - I, 'fro')

def measure_orthogonality(H: np.ndarray):
    """Log-based orthogonality measure that handles zeros and large values better"""
    col_norms = np.linalg.norm(H, axis=0, ord=2)
    
    # Filter out zero columns
    nonzero_norms = col_norms[col_norms > 1e-12]
    if len(nonzero_norms) == 0:
        return np.inf
    
    if H.shape[0] != H.shape[1]:
        _, s, _ = np.linalg.svd(H)
        # Filter out near-zero singular values
        nonzero_s = s[s > 1e-12]
        if len(nonzero_s) == 0:
            return np.inf
        log_det = np.sum(np.log(nonzero_s))
    else:
        det = np.linalg.det(H)
        if abs(det) < 1e-12:
            return np.inf
        log_det = np.log(abs(det))
    
    # Use log arithmetic to avoid overflow
    log_prod_norms = np.sum(np.log(nonzero_norms))
    return log_prod_norms - log_det

def transform(model: gp.Model):
    A = model.getA()
    for mv in model.getVars()[:model.NumVars - model.NumIntVars]:
        assert mv.VType == gp.GRB.CONTINUOUS
    
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    Ar = A[:, :model.NumVars - model.NumIntVars]
    Az = A[:, model.NumVars - model.NumIntVars:]
    Azb = sp.hstack((Az, b)).toarray().astype(np.int64, order='C')

    # om = measure_orthogonality(Azb)
    # omd = measure_orthogonality_deviation(Azb)
    # print(f"  Running LLL on {Azb.shape[0]} x {Azb.shape[1]} matrix. OM: {om}, {omd}", flush=True)
    # start = ti.default_timer()
    # rank, det, U = ntl.lll(Azb, 16, 20)
    # elapsed = ti.default_timer() - start
    # om = measure_orthogonality(Azb)
    # omd = measure_orthogonality_deviation(Azb)
    # print(f"  LLL took {elapsed:.4f} seconds, rank = {rank}, det = {det}, shape = {U.shape}, OM: {om}, {omd}")

    print(f"  Running LLL on {Azb.shape[0]} x {Azb.shape[1]} matrix.", flush=True)
    start = ti.default_timer()
    U = du.lll_fpylll_cols(Azb, 0.75, use_bkz=True)
    elapsed = ti.default_timer() - start
    print(f"  LLL took {elapsed:.4f} seconds, shape = {U.shape}")

    c = np.array(model.getAttr("Obj")).reshape((-1, 1))
    # using our knowledge of lower and upper bounds from the problem type

    model2 = gp.Model("Transformed " + model.ModelName)
    x = model2.addMVar((Ar.shape[1], 1), lb=0, vtype='C', name='x')
    y = model2.addMVar((U.shape[1], 1), lb=-gp.GRB.INFINITY, vtype='I', name='y')
    model2.setObjective(c[:x.shape[0]].T @ x + model.ObjCon, model.ModelSense)
    model2.addConstr(Ar @ x + Az @ U[0:-1, :] @ y >= b)
    model2.addConstr(-1 == U[-1, :] @ y)
    model2.addConstr(0 <= U[0:-1, :] @ y)
    model2.addConstr(U[0:-1, :] @ y <= 1)
    return model2

def main():
    instances = jl.get_instances()    
    compare_original = False
    for instance in [instances['abz5']]: # la36
        print(f"Processing instance: {instance.name}, size: {instance.jobs} x {instance.machines}")
        model = instance.as_gurobi_balas_model(use_big_m=True)
        gu.standardize_lt_to_gt(model)
        model.update()
        print(f"  Number of variables: {model.NumVars - model.NumIntVars} x {model.NumIntVars}, Number of constraints: {model.NumConstrs}")
        if compare_original:
            model.params.LogToConsole = 0
            start = ti.default_timer()
            model.optimize()
            elapsed = ti.default_timer() - start
            print(f"  Optimization time for {instance.name}: {elapsed:.4f} seconds")

        model2 = transform(model)
        model2.params.LogToConsole = 1
        # model2.params.NumericFocus = 2
        model2.update()
        start = ti.default_timer()
        model2.optimize()
        elapsed = ti.default_timer() - start
        print(f"  Post-transform optimization time for {instance.name}: {elapsed:.4f} seconds")

        if model2.status != gp.GRB.Status.OPTIMAL:
            print(f"  Model {instance.name} did not solve to optimality: {gu.status_lookup[model2.status]}")
            continue
        if round(model2.ObjVal) != instance.score:
            print(f"  Warning: Objective values differ after transformation: {model.ObjVal} vs {model2.ObjVal}")
        
if __name__ == "__main__":
    main()
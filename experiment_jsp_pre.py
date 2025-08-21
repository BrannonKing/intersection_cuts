import jsplib_loader as jl
import gurobi_utils as gu
import gurobipy as gp
import timeit as ti
import ntl_wrapper as ntl
import numpy as np
import scipy.sparse.linalg as spl

def transform(model: gp.Model):
    A = model.getA().toarray().astype(np.int64, order='C')
    # rows = []
    # for row in range(A.shape[0]):
    #     biggest = abs(A[row, :]).max()
    #     if biggest > 2:
    #         rows.append(row)
    # B = A[rows, :].toarray().astype(np.int64, order='C')
    # B = A.toarray().astype(np.int64, order='C')
    # print(f"  Running LLL on {B.shape[0]} x {B.shape[1]} matrix...", flush=True)
    # start = ti.default_timer()
    # rank, det, U = ntl.lll_left(B, 16, 20)
    # elapsed = ti.default_timer() - start
    # print(f"  LLL took {elapsed:.4f} seconds, rank = {rank}, det = {det}, shape = {U.shape}")
    U = np.eye(A.shape[0], dtype=np.int32)
    for j in range(A.shape[0] // 2):
        U[2 * j + 1, 2 * j] = 1

    assert np.linalg.det(U) == 1

    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    c = np.array(model.getAttr("Obj")).reshape((-1, 1))
    vtypes = np.array(model.getAttr("VType")).reshape((-1, 1))

    model2 = gp.Model("Transformed " + model.ModelName)
    y = model2.addMVar((A.shape[1], 1), lb=0, vtype=vtypes, name='y')
    # s = model2.addMVar((U.shape[1], 1), lb=0, vtype='C', name='s')
    model2.setObjective(c.T @ y + model.ObjCon, model.ModelSense)
    # model2.addConstr(U @ A @ y - U @ s == U @ b)  # b still has large values; need an offset
    model2.addConstr(U @ A @ y >= U @ b)  # it's just a relaxation
    model2.addConstr(A @ y >= b)  # this is the original constraint
    return model2

def main():
    instances = jl.get_instances()    
    for instance in [instances['abz5']]:
        if instance.name == 'abz7':
            break
        print(f"Processing instance: {instance.name}, size: {instance.jobs} x {instance.machines}")
        model = instance.as_gurobi_balas_model(use_big_m=True)
        gu.standardize_lt_to_gt(model)
        model.params.LogToConsole = 0
        model.update()
        print(f"  Number of variables: {model.NumVars - model.NumIntVars} x {model.NumIntVars}, Number of constraints: {model.NumConstrs}")
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
        if round(model.ObjVal) != round(model2.ObjVal):
            print(f"Warning: Objective values differ after transformation: {model.ObjVal} vs {model2.ObjVal}")
        
if __name__ == "__main__":
    main()
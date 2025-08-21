import dikin_utils as du
import gurobi_utils as gu
import numpy as np
import gurobipy as gp
import timeit as ti
import ntl_wrapper as ntl
import knapsack_loader as kl
import scipy.linalg as spl
status_lookup = {getattr(gp.GRB.Status, k): k for k in gp.GRB.Status.__dir__() if "A" <= k[0] <= "Z"}

def find_U(H: np.ndarray):
    H = H.astype(np.int64, copy=True, order='C')
    rank, det, U = ntl.lll(H, 3, 4)  # modifies H in place
    return U

def main():
    np.random.seed(42)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    for con_count in [500]:
        for var_count in [500]: #[25, 50, 100, 200]:
            print(f"Generating instances with {con_count} constraints and {var_count} variables")
            runs = 5
            instances = kl.generate(runs, con_count, var_count, 5, 10, 1000, equality=True, env=env)
            for model in instances:
                model.params.LogToConsole = 0
                # assumptions on the model: all equality constraints, fully linear objective & constraints, all vars >= 0, maximizing
                x0 = gu.relaxed_optimum(model)
                grown = gu.relax_and_grow(model, x0, 1)
                A = grown.getA().toarray()
                b = np.array(grown.getAttr("RHS")).reshape((-1, 1))
                lb = np.array(grown.getAttr("LB")).reshape((-1, 1))
                ub = np.array(grown.getAttr("UB")).reshape((-1, 1))

                # H = du.compute_H_small(np.zeros_like(ub) - 1, ub + 1, x0)
                H = du.compute_H(A, b, lb, ub, x0)
                # H = spl.sqrtm(H)
                L = np.linalg.cholesky(H)
                # assert np.allclose(L @ L.T, H), "Cholesky decomposition failed; L.T @ L != H"
                H = np.linalg.inv(L.T)
                assert np.allclose(H.imag, 0), "H has complex values; something went wrong"
                H = H.real * 1000
                Hn = np.linalg.norm(H, np.inf)
                Hd = du.orthogonality_measure_2(H)
                print(f"   H start: ||U||_inf = {Hn:.2f}, ortho = {Hd:.2f}")

                start = ti.default_timer()
                U3, mp = du.to_U_via_iteration(H)
                took = ti.default_timer() - start
                U3n = np.linalg.norm(U3, np.inf)
                U3d = du.orthogonality_measure_2(U3)
                print(f"   ItU took {took:.3f} s, ||U||_inf = {U3n:.2f}, ortho = {U3d:.2f}, iter = {mp}")

                start = ti.default_timer()
                U1 = find_U(H)
                took = ti.default_timer() - start
                U1n = np.linalg.norm(U1, np.inf)
                U1d = du.orthogonality_measure_2(U1)
                print(f"   LLL took {took:.3f} s, ||U||_inf = {U1n:.2f}, ortho = {U1d:.2f}")

                start = ti.default_timer()
                U2 = du.to_U_via_LU(H)
                took = ti.default_timer() - start
                U2n = np.linalg.norm(U2, np.inf)
                U2d = du.orthogonality_measure_2(U2)
                print(f"   LU  took {took:.3f} s, ||U||_inf = {U2n:.2f}, ortho = {U2d:.2f}")
                print()

            print()

if __name__ == "__main__":
    main()
from __future__ import annotations

import gurobi_utils as gu
import knapsack_loader as kl

if __name__ == "__main__":
    from coinor.cuppy.cuttingPlanes import gomoryMixedIntegerCut, solve
    from coinor.cuppy.milpInstance import MILPInstance

    for model in kl.generate(1, 2, 20, 5, 10, 1000, equality=True):
        A, b, c, l, u = gu.get_A_b_c_l_u(model)
        sense = ("Max", "=")
        indeces = list(range(A.shape[1]))
        m = MILPInstance(A=A, b=b.flatten(), c=c.flatten(), l=l.flatten(), u=u.flatten(), sense=sense, integerIndices=indeces, numVars=A.shape[1])
        solve(m, whichCuts=[(gomoryMixedIntegerCut, {})], display=False, debug_print=False)

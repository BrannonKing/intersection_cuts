# early thoughts:
# 1. I want to reuse the branching code on multiple solvers: Xpress, Cplex, Scip, and Gurobi (maybe).
#    That means that the branching code needs to take a generic representation of the relaxed solution.
#    And it needs to call indirectly a solver-specific method to add the branch.
#    Note: Gurobi doesn't have custom branching support; and split cuts for it won't work as I don't have the
#    tableau in the callback. That means that I have to do custom branch & bound with Gurobi. This is
#    problematic because we won't get the benefit of the other cuts, and python BnB will be slow. Let's not do this.
# 2. List of different branching mechanisms:
#    A. We need a partial solution branching mechanism. This would allow us to prune branches with a duplicate solution.
#       This means that we need to explore from s-t. But that would imply a topological order, which thing we always
#       have for any valid graph. If our graph is invalid, meaning we have partials on disjunctions, then...
#       If I wait for MIPSOL, I won't be pruning; those are all leaf nodes. It has to be MIP nodes.
#       And it can only be done on fixed variables. Hence, we could do it in the branching. That's what we want.
#       If we
# 3. Thought: we could just not branch on variables where there post-op is decided,
#       leaving them in a blurry mess that might not matter.

class Brancher:
    def make_branches(self, node_solution):
        raise NotImplementedError("")
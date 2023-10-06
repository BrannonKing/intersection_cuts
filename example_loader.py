import gurobipy as gp

import miplib_loader


def _toy2d():
    m = gp.Model("2D from bottom")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y <= 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y <= 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _toy2d_noslack():
    m = gp.Model("2D from bottom (manual slacks)")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    s1 = m.addVar(name='s1')
    s2 = m.addVar(name='s1')
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y + s1 == 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y + s2 == 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _toy2d_min():
    m = gp.Model("2D from top")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(y, sense=gp.GRB.MINIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y >= 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y >= 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _example63():
    # take from Conforti book, chapter 6

    m = gp.Model("Book example 6.3")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0

    x1 = m.addVar(name='x1', vtype=gp.GRB.BINARY)
    x2 = m.addVar(name='x2', vtype=gp.GRB.BINARY)
    x3 = m.addVar(name='x3', vtype=gp.GRB.BINARY)
    m.setObjective(0.5*x2 + x3, sense=gp.GRB.MAXIMIZE)

    m.addConstr(x1+x2+x3 <= 2, name="r0")
    m.addConstr(-x1+0.5*x3 <= 0, name="r1")
    m.addConstr(-x2+0.5*x3 <= 0, name="r2")
    m.addConstr(x1+0.5*x3 <= 1, name="r3")
    m.addConstr(-x1+x2+x3 <= 1, name="r4")

    return m


class ExampleInstance(miplib_loader.BenchmarkInstance):
    def __init__(self, model, score):
        super().__init__("=opt=", model.ModelName, score)
        self.model = model

    def as_gurobi_model(self):
        return self.model
1


def get_instances():
    return {
        "2D below": ExampleInstance(_toy2d(), 2),
        "2D above": ExampleInstance(_toy2d_min(), 3),
        "2D below (no slack)": ExampleInstance(_toy2d_noslack(), 2),
        "Book 6_3": ExampleInstance(_example63(), 0.5),
    }

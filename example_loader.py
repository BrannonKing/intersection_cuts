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


def _toy2d_slant():
    m = gp.Model("2D slanted")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(x+y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-x + 2.1 * y <= 1)  # left-hand side
    m.addConstr(2 * x -2 * y <= 3.5)  # right-hand side, meet at (1.2, 2.3)
    return m


def _toy2d_ub():
    m = gp.Model("2D from bottom (upper bounded x)")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER, ub=1.1)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y <= 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y <= 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m

def _toy2d_ub_steep():
    m = gp.Model("2D from bottom (upper bounded y)")
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER, ub=2)
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-2 * x + 0.3 * y <= -1.5)  # left-hand side
    m.addConstr(2 * x + 0.3 * y <= 3.5)  # right-hand side, meet at (1.2, 2.3)
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
    s2 = m.addVar(name='s2')
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

    x1 = m.addVar(name='x1', vtype=gp.GRB.INTEGER)
    x2 = m.addVar(name='x2', vtype=gp.GRB.INTEGER)
    x3 = m.addVar(name='x3', vtype=gp.GRB.INTEGER)
    m.setObjective(0.5*x2 + x3, sense=gp.GRB.MAXIMIZE)

    m.addConstr(x1+x2+x3 <= 2, name="r0")
    m.addConstr(x1-0.5*x3 >= 0, name="r1")
    m.addConstr(x2-0.5*x3 >= 0, name="r2")
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
        "2Dbelow": ExampleInstance(_toy2d(), 2),
        "2Dabove": ExampleInstance(_toy2d_min(), 3),
        "2Dslacks": ExampleInstance(_toy2d_noslack(), 2),
        "Book_6_3": ExampleInstance(_example63(), 0.5),  # optimum at (0, 1, 0)
        "2DbelowUBx": ExampleInstance(_toy2d_ub(), 2),
        "2DbelowUBy": ExampleInstance(_toy2d_ub_steep(), 1),
        "2Dslant": ExampleInstance(_toy2d_slant(), 3),
    }

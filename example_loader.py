import gurobipy as gp

import miplib_loader


def _toy2d(env=None):
    m = gp.Model("2D from bottom", env=env)
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

def _toy2dnoeasy(env=None):
    m = gp.Model("2D no easy cut from bottom", env=env)
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(x + y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(x + 0.3 * y <= 4.2)  # left-hand side
    m.addConstr(x + 5 * y <= 17)  # right-hand side
    return m

def _toy2d_slant(env=None):
    m = gp.Model("2D slanted", env=env)
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    m.setObjective(x+y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-x + 2.1 * y <= 1)  # left-hand side
    m.addConstr(2 * x -2 * y <= 3.5)  # right-hand side, meet at (1.2, 2.3)
    return m


def _toy2d_ub(env=None):
    m = gp.Model("2D from bottom (upper bounded x)", env=env)
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

def _toy2d_ub_steep(env=None):
    m = gp.Model("2D from bottom (steep, y<=2)", env=env)
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER, ub=2.0)
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-2 * x + 0.3 * y <= -1.5)  # left-hand side
    m.addConstr(2 * x + 0.3 * y <= 3.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _toy2d_noslack(env=None):
    m = gp.Model("2D from bottom (manual slacks)", env=env)
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    s1 = m.addVar(name='sa1')
    s2 = m.addVar(name='sa2')
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y + s1 == 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y + s2 == 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _toy2d_halfslack(env=None):
    m = gp.Model("2D from bottom (manual slack on 2)", env=env)
    m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    x = m.addVar(name='x', vtype=gp.GRB.INTEGER)
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    s2 = m.addVar(name='sa2')
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(-0.9 * x + 0.9 * y <= 1)  # left-hand side
    m.addConstr(0.9 * x + 0.6 * y + s2 == 2.5)  # right-hand side, meet at (1.2, 2.3)
    # m.addCons(1.1 * x + 0.4 * y <= 2.5)  # further right
    return m


def _toy2d_noslack_one(env=None):
    m = gp.Model("2D from bottom (manual slack, just one)", env=env)
    # m.params.Presolve = 0
    m.params.Heuristics = 0
    m.params.Cuts = 0
    y = m.addVar(name='y', vtype=gp.GRB.INTEGER)
    s1 = m.addVar(name='s1')
    m.setObjective(y, sense=gp.GRB.MAXIMIZE)

    m.addConstr(y + s1 == 2.7)
    return m


def _toy2d_min(env=None):
    m = gp.Model("2D from top", env=env)
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


def _example63(env=None):
    # take from Conforti book, chapter 6

    m = gp.Model("Book example 6.3", env=env)
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


def get_instances(env=None):
    return {
        "2Dbelow": ExampleInstance(_toy2d(env), 2),
        "2Dnoeasy": ExampleInstance(_toy2dnoeasy(env), 5),
        "2Dabove": ExampleInstance(_toy2d_min(env), 3),
        "2DbelowUBx": ExampleInstance(_toy2d_ub(env), 2),
        "2DbelowUBy": ExampleInstance(_toy2d_ub_steep(env), 1),
        "2Dslant": ExampleInstance(_toy2d_slant(env), 3),
        "Book_6_3": ExampleInstance(_example63(env), 0.5),  # optimum at (0, 1, 0)
        "2DslacksHalf": ExampleInstance(_toy2d_halfslack(env), 2),
        "2Dslacks": ExampleInstance(_toy2d_noslack(env), 2),
        "2DslackOne": ExampleInstance(_toy2d_noslack_one(env), 2),
    }

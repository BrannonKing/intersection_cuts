import gurobipy as gp
import matplotlib.axes
import matplotlib.figure
from matplotlib.patches import Circle, Arrow
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt


class PlotterBase:
    def __init__(self, model: gp.Model, var1, var2, ax: matplotlib.axes.Axes, var2_cons = None):
        self.model = model
        self.v1, self.v2 = var1, var2
        self.v2_lhs = model.getRow(var2_cons) if var2_cons is not None else None
        self.ax = ax
        self.ax.set_title(f"Model: {model.ModelName} ({var1.VarName}, {var2.VarName})")
        self.ax.set_aspect(1)
        self.ax.set_xlabel(f"{var1.VarName} ({var1.index})")
        self.ax.set_ylabel(f"{var2.VarName} ({var2.index})")
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.grid(True, which='minor', alpha=0.7)
        self.ax.grid(True, which='major', alpha=0.8, linewidth=2)
        if var1.LB > -gp.GRB.INFINITY:
            self.ax.axvline(var1.LB, color="xkcd:steel blue")
        if var1.UB < gp.GRB.INFINITY:
            self.ax.axvline(var1.UB, color="xkcd:steel blue")
        if var2.LB > -gp.GRB.INFINITY:
            self.ax.axhline(var2.LB, color="xkcd:steel blue")
        if var2.UB < gp.GRB.INFINITY:
            self.ax.axhline(var2.UB, color="xkcd:steel blue")
        self.added_lines = 0
        self.added_circles = 0

    def find_coeff_from_var(self, lhs, variable):
        for i in range(lhs.size()):
            if lhs.getVar(i).index == variable.index:
                return lhs.getCoeff(i)
        return 0.0
    
    def find_in_both(self, lhs):
        vars_in_lhs = [(i, lhs.getVar(i)) for i in range(lhs.size()) if lhs.getVar(i).index not in [self.v1.index, self.v2.index]]
        indexes = set(v.index for _, v in vars_in_lhs)
        v2idx = -1
        v2i = -1
        for i in range(self.v2_lhs.size()):
            idx = self.v2_lhs.getVar(i).index
            if idx in indexes:
                v2idx = idx
                v2i = i
                break
        if v2idx >= 0:  # TODO: write this in a more efficient way
            for i, v in vars_in_lhs:
                if v.index == v2idx:
                    return i, v2i
        return -1, v2i


    def find_coeffs(self, lhs):
        # if our v2 is actually the objective variable
        # we should not let it be zero; we should solve for it in terms of other variables
        a = self.find_coeff_from_var(lhs, self.v1)
        b = self.find_coeff_from_var(lhs, self.v2)

        # handle the situation where our constraint lacks v2 but has some variables in our objective equality constr
        # algorithm:
        # 1. if v2 not in constraint
        # 2. pick a var != v1 or v2 and in both lhs and v2_lhs
        # 3. b = var's lhs.coef * -v2's v2_lhs.coef
        # 4. a -= var's lhs.coef * v1's v2_lhs.coef
        if b == 0.0 and self.v2_lhs is not None:
            i1, i2 = self.find_in_both(lhs)  # e.g. y in both our objective and constraint
            if i1 >= 0:
                lg1, lg2 = lhs.getCoeff(i1), self.v2_lhs.getCoeff(i2)
                a -= lg1 * self.find_coeff_from_var(self.v2_lhs, self.v1) / lg2
                b = -lg1 * self.find_coeff_from_var(self.v2_lhs, self.v2) / lg2

        return a, b

    def add_constraint(self, constraint, color='xkcd:gold'):
        assert isinstance(constraint, (gp.Constr, gp.MConstr))
        lhs, rhs = self.model.getRow(constraint), constraint.RHS
        cof1, cof2 = self.find_coeffs(lhs)
        if cof1 == 0.0 and cof2 == 0.0:
            return
        if cof1 == 0.0:
            self.ax.axhline(rhs / cof2, color=color)
        elif cof2 == 0.0:
            self.ax.axvline(rhs / cof1, color=color)
        else:
            slope = -cof1 / cof2
            c1 = (0, rhs / cof2)
            self.ax.axline(c1, slope=slope, color=color)
        self.added_lines += 1
        # TODO: add line label, both name and count
        # TODO: get the arrows one them for the sense
        # TODO: darken each line a little more than the previous

        # if constraint.Sense == '>':
        #     self.ax.arrow(c2[0], c2[1], -.1 * (c2[1] - c1[1]), .1 * (c2[0] - c1[0]),
        #                     head_width=0.05, head_length=0.1, color='gold')
        # else:
        #     self.ax.arrow(c2[0], c2[1], .1 * (c2[1] - c1[1]), -.1 * (c2[0] - c1[0]),
        #                     head_width=0.05, head_length=0.1, color='gold')

    def add_ball(self, radius):
        p1 = self.v1.X
        p2 = self.v2.X
        if len(self.ax.patches) > 0:
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
        else:
            xl = (p1, p1)
            yl = (p2, p2)
        if radius > 2 or xl[1] - xl[0] > 10 or yl[1] - yl[0] > 7:
            self.ax.minorticks_off()
        scl = 2.5
        self.ax.set_xlim(min(xl[0], p1 - radius*scl), max(xl[1], p1 + radius*scl))
        self.ax.set_ylim(min(yl[0], p2 - radius*scl), max(yl[1], p2 + radius*scl))
        self.added_circles += 1
        circle = Circle((p1, p2), radius, color=(1 / self.added_circles, 0, 0), fill=False)
        self.ax.add_patch(circle)

    def render(self):
        plt.show()


class Plotter2D(PlotterBase):
    def __init__(self, model):
        assert model.NumIntVars == 2
        self.fig = plt.figure(dpi=96, figsize=(7, 7), layout="constrained")
        variables = [v for v in model.getVars() if v.VType != 'C']
        super().__init__(model, variables[0], variables[1], self.fig.add_subplot())
        for c in model.getConstrs():
            self.add_constraint(c, "xkcd:medium blue")


class Plotter3D:
    def __init__(self, model):
        # TODO: switch this to use itertools.combinations
        assert model.NumIntVars == 3
        self.fig, self.axs = plt.subplots(3, 1, dpi=96, figsize=(7, 21), layout="constrained")
        variables = [v for v in model.getVars() if v.VType != 'C']
        self.p1 = PlotterBase(model, variables[0], variables[1], self.axs[0])
        self.p2 = PlotterBase(model, variables[0], variables[2], self.axs[1])
        self.p3 = PlotterBase(model, variables[1], variables[2], self.axs[2])
        for c in model.getConstrs():
            self.p1.add_constraint(c, "xkcd:medium blue")
            self.p2.add_constraint(c, "xkcd:medium blue")
            self.p3.add_constraint(c, "xkcd:medium blue")

    def add_constraint(self, constraint, color='xkcd:gold'):
        self.p1.add_constraint(constraint, color)
        self.p2.add_constraint(constraint, color)
        self.p3.add_constraint(constraint, color)
    
    def add_ball(self, radius):
        self.p1.add_ball(radius)
        self.p2.add_ball(radius)
        self.p3.add_ball(radius)

    def render(self):
        plt.show()


class PlotterObjective:
    def __init__(self, model, objective_var, objective_cons):
        nv = min(10, model.NumIntVars)
        if objective_var.VType != 'C':
            nv -= 1
        self.fig, self.axs = plt.subplots(nv, 1, dpi=96, figsize=(7, 7*nv), layout="constrained")
        self.fig.tight_layout(pad=3)
        if nv == 1:
            self.axs = [self.axs]
        variables = [v for v in model.getVars() if v.VType != 'C' and v.index != objective_var.index]
        self.ps = [PlotterBase(model, v, objective_var, self.axs[i], objective_cons) for i, v in enumerate(variables[:nv])]
        for c in model.getConstrs():
            for p in self.ps:
                p.add_constraint(c, "xkcd:medium blue")

    def add_constraint(self, constraint, color='xkcd:gold'):
        for p in self.ps:
            p.add_constraint(constraint, color)
    
    def add_ball(self, radius):
        for p in self.ps:
            p.add_ball(radius)

    def render(self):
        plt.show()


def create(model: gp.Model, objective_var=None, objective_constraint=None):
    model.update()
    # maybe find the integer/binary vars only?
    if objective_var is not None:
        return PlotterObjective(model, objective_var, objective_constraint)
    elif model.NumIntVars == 2:
        return Plotter2D(model)
    if model.NumIntVars == 3:
        return Plotter3D(model)
    return None

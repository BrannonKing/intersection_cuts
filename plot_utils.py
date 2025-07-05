import gurobipy as gp
import numpy as np
import matplotlib.axes
import matplotlib.figure
from matplotlib.patches import Circle, Arrow
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt
import seaborn as sns


class PlotterBase:
    def __init__(self, model: gp.Model, var1, var2, ax: matplotlib.axes.Axes, var2_cons=None):
        self.model = model
        self.v1, self.v2 = var1, var2
        self.v2_lhs = model.getRow(var2_cons) if var2_cons is not None else None
        self.ax = ax
        self.ax.set_title(f"Model: {model.ModelName} ({var1.VarName}, {var2.VarName})")
        self.ax.set_xlabel(f"{var1.VarName} ({var1.index})")
        self.ax.set_ylabel(f"{var2.VarName} ({var2.index})")
        # self.ax.xaxis.set_major_locator(MultipleLocator(1))
        # self.ax.yaxis.set_major_locator(MultipleLocator(1))
        # self.ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        # self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.grid(True, which='minor', alpha=0.7)
        self.ax.grid(True, which='major', alpha=0.8, linewidth=2)
        if var1.LB > -gp.GRB.INFINITY:
            self.ax.axvline(var1.LB, color="xkcd:black")
        if var1.UB < gp.GRB.INFINITY:
            self.ax.axvline(var1.UB, color="xkcd:black")
        if var2.LB > -gp.GRB.INFINITY:
            self.ax.axhline(var2.LB, color="xkcd:black")
        if var2.UB < gp.GRB.INFINITY:
            self.ax.axhline(var2.UB, color="xkcd:black")
        self.added_lines = 0
        self.added_circles = 0
        self.ax.set_aspect(1)

    def find_coeff_from_var(self, lhs, variable):
        s = 0.0
        for i in range(lhs.size()):
            if lhs.getVar(i).index == variable.index:
                s += lhs.getCoeff(i)
        return s
    
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
        # note: this is wholly insufficient. If there is a slack variable in this,
        # it should be resolved in terms of other vars. and that is true for manual slacks as well.
        lhs, rhs = self.model.getRow(constraint), constraint.RHS
        cof1, cof2 = self.find_coeffs(lhs)
        if cof1 == 0.0 and cof2 == 0.0:
            return
        # TODO: ask Robert how to draw the objective constraint
        offset = 0.1 if constraint.Sense == '>' else -0.1
        sign = lambda a: bool(a > 0) - bool(a < 0)
        if cof1 == 0.0:
            self.ax.axhline(rhs / cof2, color=color)
            if constraint.Sense != '=':
                self.ax.axhline(rhs / cof2 + sign(cof2) * offset, color=color, alpha=0.3, linestyle='--')
        elif cof2 == 0.0:
            self.ax.axvline(rhs / cof1, color=color)
            if constraint.Sense != '=':
                self.ax.axvline(rhs / cof1 + sign(cof1) * offset, color=color, alpha=0.3, linestyle='--')
        else:
            slope = -cof1 / cof2
            c1 = (0, rhs / cof2)
            self.ax.axline(c1, slope=slope, color=color)
            if constraint.Sense != '=':
                self.ax.axline((c1[0] + sign(cof1) * offset, c1[1] + sign(cof2) * offset), slope=slope, color=color, alpha=0.3, linestyle='--')
        self.added_lines += 1
        # TODO: add line label, both name and count
        # TODO: get the arrows on them for the sense
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
            xl = (0, p1)
            yl = (0, p2)
        if radius > 2 or xl[1] - xl[0] > 10 or yl[1] - yl[0] > 7:
            self.ax.minorticks_off()
        scl = 1.1
        self.ax.set_xlim(min(xl[0], p1 - radius*scl), max(xl[1], p1 + radius*scl))
        self.ax.set_ylim(min(yl[0], p2 - radius*scl), max(yl[1], p2 + radius*scl))
        self.added_circles += 1
        circle = Circle((p1, p2), radius, color=(1 / self.added_circles, 0, 0), alpha=0.0, fill=False)
        self.ax.add_patch(circle)
        self.ax.plot(p1, p2, 'ro')

    def add_point(self, point, color='xkcd:orange'):
        x = point[self.v1.index]
        y = point[self.v2.index]
        self.ax.plot(x, y, 'o', color=color)

    def render(self):
        plt.show()


class Plotter2D(PlotterBase):
    def __init__(self, model):
        assert model.NumIntVars == 2 or model.NumVars == 2
        self.fig = plt.figure(dpi=96, figsize=(10, 10), layout="constrained")
        variables = [v for v in model.getVars() if v.VType != 'C' or model.NumVars == 2]
        super().__init__(model, variables[0], variables[1], self.fig.add_subplot())
        for c in model.getConstrs():
            self.add_constraint(c, "xkcd:medium blue")


class Plotter3D:
    def __init__(self, model):
        # TODO: switch this to use itertools.combinations
        assert model.NumIntVars == 3
        self.fig, self.axs = plt.subplots(3, 1, dpi=96, figsize=(10, 30), layout="constrained")
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
        nv = min(30, model.NumIntVars)
        if objective_var.VType != 'C':
            nv -= 1
        self.fig, self.axs = plt.subplots(nv, 1, dpi=96, figsize=(10, 10*nv), layout="constrained")
        # self.fig.tight_layout(pad=3)
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
    elif model.NumIntVars == 2 or model.NumVars == 2:
        return Plotter2D(model)
    if model.NumIntVars == 3:
        return Plotter3D(model)
    return None

def plot_constraints_lte(title, A, b, l, u, senses, x_bounds=(-1, 7), y_bounds=(-1, 7), points=None, fig=None):
    """
    Plots the feasible region defined by Ax <= b for 2D constraints.

    Parameters:
    - A: Coefficient matrix (m x 2) for m constraints.
    - b: Right-hand side vector (m x 1).
    - x_bounds: Tuple defining the x-axis range (min_x, max_x).
    - y_bounds: Tuple defining the y-axis range (min_y, max_y).
    - points: Array of points to plot as red dots (n x 2).
    """
    x = np.linspace(x_bounds[0], x_bounds[1], 500)
    
    fig, ax = plt.subplots(figsize=(8, 8)) if fig is None else (fig, fig.gca())
    colors = ['DeepSkyBlue', 'LimeGreen', 'DarkOrange', 'DarkViolet', 'Crimson', 'DarkSlateGray']

    # Compute the feasible region
    # y = np.linspace(y_bounds[0], y_bounds[1], 500)
    # X, Y = np.meshgrid(x, y)
    # Z = np.ones_like(X, dtype=bool)
    # for i in range(A.shape[0]):
    #     Z &= (A[i, 0] * X + A[i, 1] * Y <= b[i])

    # ax.contourf(X, Y, Z, levels=1, colors=["lightblue"], alpha=0.3)
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)
    b = b.flatten()
    l = l.flatten()
    u = u.flatten()
    ai = 0

    for i, lwr in enumerate(l[:2]):
        if i == 0 and lwr > x_bounds[0] and lwr != 0:
            ax.axvline(lwr, label=f"x >= {lwr:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1
        elif i == 1 and lwr > y_bounds[0] and lwr != 0:
            ax.axhline(lwr, label=f"y >= {lwr:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1

    for i, upr in enumerate(u[:2]):
        if i == 0 and upr < x_bounds[1] and upr != 0:
            ax.axvline(upr, label=f"x <= {upr:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1
        elif i == 1 and upr < y_bounds[1] and upr != 0:
            ax.axhline(upr, label=f"y <= {upr:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1
    
    # Plot the constraint lines
    for i in range(A.shape[0]):
        if A[i, 1] != 0:
            y_constraint = (b[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, y_constraint, label=f"{A[i,0]:.3f}x + {A[i,1]:.3f}y {senses[i]}= {b[i]:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1
        else:
            x_constraint = b[i] / A[i, 0]
            ax.axvline(x=x_constraint, label=f"x {senses[i]}= {x_constraint:.3f}", linewidth=2, color=colors[ai % len(colors)])
            ai += 1

    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot additional points if provided
    if points is not None:
        points = np.array(points)
        if points.shape[1] < 2:
            points = np.hstack([points, np.zeros((points.shape[0], 1))])
        ax.scatter(points[:, 0], points[:, 1], color='goldenrod', label='Points')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_title(title)
    return fig
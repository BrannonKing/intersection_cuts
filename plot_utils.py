import gurobipy as gp
import matplotlib.axes
import matplotlib.figure
from matplotlib.patches import Circle, Arrow
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt


class PlotterBase:
    def __init__(self, model: gp.Model, i1, i2, ax: matplotlib.axes.Axes):
        self.model = model
        self.i1, self.i2 = i1, i2
        self.ax = ax
        self.ax.set_title(f"Model: {model.ModelName} ({i1}, {i2})")
        self.ax.set_aspect(1)
        self.ax.set_xlabel(str(i1))
        self.ax.set_ylabel(str(i2))
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        self.ax.grid(True, which='minor', alpha=0.7)
        self.ax.grid(True, which='major', alpha=0.8, linewidth=2)
        self.added_lines = 0
        self.added_circles = 0

    def find_indexes(self, lhs):
        a, b = 0.0, 0.0
        for i in range(lhs.size()):
            if lhs.getVar(i).index == self.i1:
                a = lhs.getCoeff(i)
            elif lhs.getVar(i).index == self.i2:
                b = lhs.getCoeff(i)
        return a, b

    def add_constraint(self, constraint, color='orange'):
        assert isinstance(constraint, (gp.Constr, gp.MConstr))
        lhs, rhs = self.model.getRow(constraint), constraint.RHS
        cof1, cof2 = self.find_indexes(lhs)
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

    def add_ball(self, point, radius):
        if len(self.ax.patches) > 0:
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
        else:
            xl = (point[self.i1], point[self.i1])
            yl = (point[self.i2], point[self.i2])
        self.ax.set_xlim(min(xl[0], point[self.i1] - radius*1.5), max(xl[1], point[self.i1] + radius*1.5))
        self.ax.set_ylim(min(yl[0], point[self.i2] - radius*1.5), max(yl[1], point[self.i2] + radius*1.5))
        self.added_circles += 1
        circle = Circle((point[self.i1], point[self.i2]), radius, color=(1 / self.added_circles, 0, 0), fill=False)
        self.ax.add_patch(circle)

    def render(self):
        plt.show()


class Plotter2D(PlotterBase):
    def __init__(self, model):
        self.fig = plt.figure(dpi=96, figsize=(7,7), layout="constrained")
        super().__init__(model, 0, 1, self.fig.add_subplot())
        for c in model.getConstrs():
            self.add_constraint(c, "cyan")


class Plotter3D:
    def __init__(self, model):
        # TODO: switch this to use itertools.combinations
        self.fig, self.axs = plt.subplots(3, 1, dpi=96, figsize=(7,21), layout="constrained")
        self.p1 = PlotterBase(model, 0, 1, self.axs[0])
        self.p2 = PlotterBase(model, 0, 2, self.axs[1])
        self.p3 = PlotterBase(model, 1, 2, self.axs[2])
        for c in model.getConstrs():
            self.p1.add_constraint(c, "cyan")
            self.p2.add_constraint(c, "cyan")
            self.p3.add_constraint(c, "cyan")

    def add_constraint(self, constraint, color='orange'):
        self.p1.add_constraint(constraint, color)
        self.p2.add_constraint(constraint, color)
        self.p3.add_constraint(constraint, color)
    
    def add_ball(self, point, radius):
        self.p1.add_ball(point, radius)
        self.p2.add_ball(point, radius)
        self.p3.add_ball(point, radius)

    def render(self):
        plt.show()


def create(model: gp.Model):
    model.update()
    # maybe find the integer/binary vars only?
    if model.NumVars == 2:
        return Plotter2D(model)
    if model.NumVars == 3:
        return Plotter3D(model)
    return None

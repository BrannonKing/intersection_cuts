import gurobipy as gp
import matplotlib.axes
import matplotlib.figure
from matplotlib.lines import Line2D, _AxLine
from matplotlib.patches import Circle, Arrow
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        self.model = None
        self.fig: matplotlib.figure.Figure = plt.figure(dpi=96)  # figsize=(10, 10), 
        self.ax: matplotlib.axes.Axes = self.fig.add_subplot()
        self.ax.set_aspect(1)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.grid(True, alpha=0.6)
        self.added_lines = 0
        self.added_circles = 0

    def add_constraint(self, constraint):
        assert isinstance(constraint, gp.Constr)
        lhs, rhs = self.model.getRow(constraint), constraint.RHS
        # cx = (0, rhs / lhs.getCoeff(0))
        # cy = (rhs / lhs.getCoeff(1), 0)
        # self.ax.add_line(Line2D(cx, cy, color='orange'))
        c1 = (0, rhs / lhs.getCoeff(1))
        c2 = (rhs / lhs.getCoeff(0), 0)
        line = _AxLine(c1, c2, None, color='orange')
        self.ax.add_line(line)
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

    def set_model(self, model):
        self.model = model
        self.ax.set_title(model.ModelName)
        for c in model.getConstrs():
            self.add_constraint(c)

    def add_ball(self, point, radius):
        if len(self.ax.patches) > 0:
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
        else:
            xl = (point[0], point[0])
            yl = (point[1], point[1])
        self.ax.set_xlim(min(xl[0], point[0] - radius), max(xl[1], point[0] + radius))
        self.ax.set_ylim(min(yl[0], point[1] - radius), max(yl[1], point[1] + radius))
        self.added_circles += 1
        color = 1 - 1 / self.added_circles
        circle = Circle(point, radius, color=(1, color, color), fill=False)
        self.ax.add_patch(circle)

    def render(self):
        plt.show()


def create(model: gp.Model):
    model.update()
    if model.NumVars > 2:
        return None

    p = Plotter()
    p.set_model(model)
    return p
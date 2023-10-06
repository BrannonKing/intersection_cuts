import gurobipy as gp
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.model = None
        self.fig: matplotlib.figure.Figure = plt.figure(figsize=(10, 10), dpi=96)
        self.ax: matplotlib.axes.Axes = self.fig.add_subplot()
        self.ax.set_aspect(1)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.grid(True, alpha=0.6)

    def add_constraint(self, constraint: gp.Constr):
        lhs, rhs = self.model.getRow(constraint), constraint.RHS
        lhs.
        for (c1, c2, sense) in cuts:
            self.ax.axline(c1, c2, color='orange')
            if constraint.Sense == '<':
                self.ax.arrow(c2[0], c2[1], -.1 * (c2[1] - c1[1]), .1 * (c2[0] - c1[0]),
                              head_width=0.05, head_length=0.1, color='gold')
            else:
                self.ax.arrow(c2[0], c2[1], .1 * (c2[1] - c1[1]), -.1 * (c2[0] - c1[0]),
                              head_width=0.05, head_length=0.1, color='gold')

    def set_model(self, model):
        self.model = model
        for c in model.getConstrs():
            self.add_constraint(c)

    def add_ball(self, point, radius):
        circle = plt.Circle(point, radius, color='r', )
        self.ax.add_patch(circle)

    def render(self):
        self.fig.show()


def create(model: gp.Model):
    model.update()
    if model.NumVars > 2:
        return None

    p = Plotter()
    p.set_model(model)
    return p
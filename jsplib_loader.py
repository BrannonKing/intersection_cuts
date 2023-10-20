import gurobipy as gp
import json
import pathlib

from miplib_loader import BenchmarkInstance


class JspInstance(BenchmarkInstance):
    def __init__(self, folder, definition):
        self.folder = folder
        self.name = definition['name']
        self.jobs = definition['jobs']
        self.machines = definition['machines']
        self.optimum = definition.get('optimum', None)
        self.known_optimum = self.optimum is not None
        self.path = definition['path']
        self.work = None

    def _ensure_work_loaded(self):
        if self.work is not None:
            return
        jobs = []
        with open(self.folder / self.path) as file_handle:
            lines = file_handle.readlines()[-self.jobs:]
            for line in lines:
                splits = line.split()
                operations = [(int(splits[i]), int(splits[i+1])) for i in range(0, len(splits), 2)]
                jobs.append(operations)
        self.work = jobs

    def as_gurobi_balas_model(self, use_big_m):
        # this comes from the original Balas PERT paper, but it's usually called the Manne Model,
        # although the Manne Model is usually written in a convoluted way with sums used to find a single value
        self._ensure_work_loaded()
        model = gp.Model(self.name)
        J = self.jobs
        M = self.machines
        O = M  # assuming one operation per job for each machine

        s = model.addMVar((J, O), vtype='C', name='s')  # start time for each task t on job j
        x = model.addMVar((M, J, J), vtype='B', name='x')  # task a happens before task b (or not)
        cmax = model.addVar(name='c_max')
        model.setObjective(cmax, gp.GRB.MINIMIZE)

        bigM = 0
        lookup = {m: [] for m in range(M)}
        for j, job in enumerate(self.work):
            for o, (m, d) in enumerate(job):  # operation, (machine, delay)
                model.addConstr((s[j, o+1] if o+1 < O else cmax) - s[j, o] >= d, f"order_{j}_{o}")
                lookup[m].append((j, o, d))
                bigM += d

        for m in range(M):
            for l1, (j1, o1, d1) in enumerate(lookup[m]):
                for l2, (j2, o2, d2) in enumerate(lookup[m][l1+1:], l1+1):
                    if use_big_m:
                        model.addConstr(s[j2, o2] >= d1 + s[j1, o1] - bigM * (1 - x[m, l1, l2]))
                        model.addConstr(s[j1, o1] >= d2 + s[j2, o2] - bigM * x[m, l1, l2])
                    else:
                        # quadratic constraints:
                        # model.addConstr(s[j2, o2] >= (d1 + s[j1, o1]) * x[m, l1, l2])
                        # model.addConstr(s[j1, o1] >= (d2 + s[j2, o2]) * (1-x[m, l1, l2]))
                        # these are the fastest approach:
                        model.addGenConstrIndicator(x[m, l1, l2], True, s[j2, o2] >= d1 + s[j1, o1])
                        model.addGenConstrIndicator(x[m, l1, l2], False, s[j1, o1] >= d2 + s[j2, o2])

        model.params.AggFill = 10  # from model.tune()
        model.params.GomoryPasses = 1
        return model

    def as_gurobi_model(self):
        return self.as_gurobi_balas_model(use_big_m=True)


def get_instances():
    path = pathlib.Path('JSPLIB/instances.json')
    if not path.exists():
        path = pathlib.Path('../JSPLIB/instances.json')
        if not path.exists():
            print("Please run this one-time command: git clone https://github.com/tamy0612/JSPLIB")
            return None
    with open(path, "rt") as file_handle:
        instances = json.load(file_handle)

    return {instance['name']: JspInstance(path.parent, instance) for instance in instances}

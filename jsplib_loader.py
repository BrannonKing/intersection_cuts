import json
import pathlib
from miplib_loader import BenchmarkInstance


class JspInstance(BenchmarkInstance):
    def __init__(self, folder, definition):
        self.folder = folder
        self.name = definition['name']
        self.jobs = definition['jobs']
        self.machines = definition['machines']
        self.score = definition.get('optimum', None)
        self.known_optimum = self.score is not None
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

    def as_gurobi_balas_model(self, use_big_m, use_n11=False, env=None):
        import gurobipy as gp
        # this comes from the original Balas PERT paper, but it's usually called the Manne Model,
        # although the Manne Model is usually written in a convoluted way with sums used to find a single value
        self._ensure_work_loaded()
        model = gp.Model(self.name, env=env)
        J = self.jobs
        M = self.machines
        O = M  # assuming one operation per job for each machine

        s = model.addMVar((J, O), vtype='C', name='s')  # start time for each task t on job j
        if use_n11:
            x = model.addMVar((M, J, J), vtype='C', name='x', lb=-1, ub=1)  # task a happens before task b (or not)
        else:
            x = model.addMVar((M, J, J), vtype='B', name='x')  # task a happens before task b (or not)
        cmax = model.addVar(name='c_max')
        model.setObjective(cmax, gp.GRB.MINIMIZE)

        bigM = 1
        lookup = {m: [] for m in range(M)}
        for j, job in enumerate(self.work):
            for o, (m, d) in enumerate(job):  # operation, (machine, delay)
                model.addConstr((s[j, o+1] if o+1 < O else cmax) - s[j, o] >= d, f"order_{j}_{o}")
                lookup[m].append((j, o, d))
                bigM += d

        for m in range(M):
            for l1, (j1, o1, d1) in enumerate(lookup[m]):
                for l2, (j2, o2, d2) in enumerate(lookup[m][l1+1:], l1+1):
                    if use_big_m or use_n11:
                        model.addConstr(s[j2, o2] >= d1 + s[j1, o1] - bigM * (1 - x[m, l1, l2]))
                        model.addConstr(s[j1, o1] >= d2 + s[j2, o2] - bigM * (x[m, l1, l2] + (1 if use_n11 else 0)))
                    else:
                        # quadratic constraints:
                        # model.addConstr(s[j2, o2] >= (d1 + s[j1, o1]) * x[m, l1, l2])
                        # model.addConstr(s[j1, o1] >= (d2 + s[j2, o2]) * (1-x[m, l1, l2]))
                        # these are the fastest approach:
                        model.addGenConstrIndicator(x[m, l1, l2], True, s[j2, o2] >= d1 + s[j1, o1])
                        model.addGenConstrIndicator(x[m, l1, l2], False, s[j1, o1] >= d2 + s[j2, o2])
        
        if use_n11:
            model.addConstr(x*x == 1)
            # nm = model.addVar(name='nm', lb=M*J*J, vtype='C')
            # model.addConstr(nm == gp.norm(x.reshape((M*J*J,)), 1.0))

        model.params.AggFill = 10  # from model.tune()
        model.params.GomoryPasses = 1
        model._s = s
        model._x = x
        return model

    def as_gurobi_balas_equality_model(self, env=None):
        import gurobipy as gp
        # this comes from the original Balas PERT paper, but it's usually called the Manne Model,
        # although the Manne Model is usually written in a convoluted way with sums used to find a single value
        self._ensure_work_loaded()
        model = gp.Model(self.name, env=env)
        J = self.jobs
        M = self.machines
        O = M  # assuming one operation per job for each machine

        s = model.addMVar((J, O), vtype='C', name='s')  # start time for each task t on job j
        x = model.addMVar((M, J, J), vtype='B', name='x')  # task a happens before task b (or not)
        cmax = model.addVar(name='c_max')
        model.setObjective(cmax, gp.GRB.MINIMIZE)

        bigM = 1
        lookup = {m: [] for m in range(M)}
        for j, job in enumerate(self.work):
            for o, (m, d) in enumerate(job):  # operation, (machine, delay)
                slack = model.addVar(name=f"slack_{j}_{o}", vtype='I')
                model.addConstr((s[j, o+1] if o+1 < O else cmax) - s[j, o] - slack == d, f"order_{j}_{o}")
                lookup[m].append((j, o, d))
                bigM += d

        for m in range(M):
            for l1, (j1, o1, d1) in enumerate(lookup[m]):
                for l2, (j2, o2, d2) in enumerate(lookup[m][l1+1:], l1+1):
                    slack1 = model.addVar(name=f"slack1_{m}_{l1}_{l2}", vtype='I')
                    model.addConstr(s[j2, o2] - s[j1, o1] + bigM * (1 - x[m, l1, l2]) - slack1 == d1)
                    slack2 = model.addVar(name=f"slack2_{m}_{l1}_{l2}", vtype='I')  # TODO: figure out relation between slack1 and slack2
                    model.addConstr(s[j1, o1] - s[j2, o2] + bigM * (x[m, l1, l2]) - slack2 == d2)
        
        model.params.AggFill = 10  # from model.tune()
        model.params.GomoryPasses = 1
        model._s = s
        model._x = x
        return model

    def as_gurobi_model(self):
        return self.as_gurobi_balas_model(use_big_m=True)

    # def adjacency_matrix(self):
    #     self._ensure_work_loaded()
    #
    # def score(self, int_vars, int_idxj, nearest, tol=1e-5):
    #     a = self.adjacency_matrix()
    #     for var in int_vars:
    #         if var.X < tol:
    #             a[]

    def as_cplex_balas_model(self, use_big_m, brancher=None):
        self._ensure_work_loaded()

        import cplex as cp

        class CplexBrancherCB(cp.callbacks.BranchCallback):
            def __init__(self, env):
                super().__init__(env)
                self._brancher = None

            def set_brancher(self, brancher):
                self._brancher = brancher

            def __call__(self, *args, **kwargs):
                if self.get_branch_type() != cp.callbacks.BranchCallback.branch_type.variable:
                    return

                vals = self.get_values()
                feas = self.get_feasibilities()

                branches = self._brancher(vals, feas)
                if branches is None:
                    self.prune()
                    return

                for branch in branches:
                    variable = (branch.variable_idx, 'L' if branch.is_lower else 'U', branch.variable_value)
                    self.make_branch(branch.score, [variable], node_data=(variable[0], self.get_objective_value()))

        # class CplexMipNodeCB(cp.callbacks.CutCallback):
        #     def __call__(self, *args, **kwargs):
        #         self._get_node_info()


        model = cp.Cplex()
        J = self.jobs
        M = self.machines
        # if more jobs than machines, we have clique
        O = M  # assuming one operation per job for each machine

        # this comes from the original Balas PERT paper, but it's usually called the Manne Model,
        # although the Manne Model is usually written in a convoluted way with sums used to find a single value
        model.variables.add(lb=[0.0]*J*O)
        cmax = model.variables.add(obj=[1.0], lb=[0.0], names=['c_max'])[0]

        # Y = np.zeros((J*M, J*M+1), dtype=np.int32)
        lookup = {m: [] for m in range(M)}
        bigM = 1
        for j, job in enumerate(self.work):
            for o, (m, d) in enumerate(job):  # operation, (machine, delay)
                model.linear_constraints.add([cp.SparsePair([j*O+o+1 if o + 1 < O else cmax, j*O+o], [1.0, -1.0])],
                                              senses=['G'], rhs=[d])
                lookup[m].append((j, o, d))
                bigM += d

        for m in range(M):
            for l1, (j1, o1, d1) in enumerate(lookup[m]):
                for l2, (j2, o2, d2) in enumerate(lookup[m][l1 + 1:], l1 + 1):
                    bv = model.variables.add(types=[model.variables.type.binary])[0]
                    if use_big_m:
                        model.linear_constraints.add([cp.SparsePair([j2*O+o2, j1*O+o1, bv], [1.0, -1.0, -bigM]),
                                                      cp.SparsePair([j1*O+o1, j2*O+o2, bv], [1.0, -1.0, bigM])],
                                                     senses=['G', 'G'], rhs=[d1 - bigM, d2])
                    else:
                        model.indicator_constraints.add(cp.SparsePair([j2*O+o2, j1*O+o1], [1.0, -1.0]), rhs=d1, sense='G', indvar=bv, complemented=0)
                        model.indicator_constraints.add(cp.SparsePair([j1*O+o1, j2*O+o2], [1.0, -1.0]), rhs=d2, sense='G', indvar=bv, complemented=1)

        model._instance = self

        if brancher is not None:
            cb = model.register_callback(self.CplexBrancherCB)
            cb.set_brancher(brancher)

            #cb2 = model.register_callback(self.)

        return model

    def as_scip_balas_model(self, use_big_m, brancher=None):
        self._ensure_work_loaded()
        import pyscipopt as scip

        class ScipBrancherCB(scip.Branchrule):
            def __init__(self, brancher):
                super().__init__()
                self._brancher = brancher

            def branchexecext(self, allowaddcons):
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            def branchexecps(self, allowaddcons):
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            def branchexeclp(self, allowaddcons):
                m: scip.Model = self.model
                cand_vars, *_ = m.getLPBranchCands()
                #  lpcands: list of variables of LP branching candidates
                #  lpcandssol: list of LP candidate solution values
                #  lpcandsfrac	list of LP candidate fractionalities
                #  nlpcands:    number of LP branching candidates
                #  npriolpcands: number of candidates with maximal priority
                #  nfracimplvars: number of fractional implicit integer variables
                worst_var = None
                worst_gap = 0
                db = m.getDualbound()
                for i, cand in enumerate(cand_vars):
                    splits = cand.name.split('/')
                    assert len(splits) == 4  # and splits[0] == 'x' or 't_x'
                    machine, j1, j2 = int(splits[1]), int(splits[2]), int(splits[3])
                    o1, d1 = self.tasks[machine, j1]
                    o2, d2 = self.tasks[machine, j2]
                    # if d1 + d2 > worst_gap:
                    #     worst_gap = d1 + d2
                    #     worst_var = cand
                    # continue

                    v1 = self.s[j1, o1].getLPSol()
                    v2 = self.s[j2, o2].getLPSol()

                    # if (o1 == 9 and v1 + d1 <= db) or (o2 == 9 and v2 + d2 <= db):
                    #     worst_var = cand
                    #     break

                    if v2 <= v1 < v2 + d2:
                        gap = v2 + d2 - v1
                    elif v1 <= v2 < v1 + d1:
                        gap = v1 + d1 - v2
                    else:
                        continue
                    if gap > worst_gap:
                        worst_gap = gap
                        worst_var = cand

                if worst_var is not None:
                    left, _, right = m.branchVar(worst_var)

                    # could call m.updateNodeLowerbound(left) if we knew its lower bound
                    return {"result": scip.SCIP_RESULT.BRANCHED}

                return {"result": scip.SCIP_RESULT.DIDNOTFIND}


        model = scip.Model(problemName=self.name)
        J = self.jobs
        M = self.machines
        # if more jobs than machines, we have clique
        O = M  # assuming one operation per job for each machine

        # this comes from the original Balas PERT paper, but it's usually called the Manne Model,
        # although the Manne Model is usually written in a convoluted way with sums used to find a single value
        s = {(j, o): model.addVar(vtype='C', name=f's/{j}/{o}') for j in range(J) for o in range(O)}

        # task 1 happens before task 2:
        cmax = model.addVar(vtype='C', name='c_max')
        model.setObjective(cmax)

        # Y = np.zeros((J*M, J*M+1), dtype=np.int32)
        lookup = {m: [] for m in range(M)}
        bigM = 1
        for j, job in enumerate(self.work):
            for o, (m, d) in enumerate(job):  # operation, (machine, delay)
                model.addCons((s[j, o + 1] if o + 1 < O else cmax) - s[j, o] >= d, name=f'jo/{j}/{o}')
                lookup[m].append((j, o, d))
                bigM += d

        for m in range(M):
            for l1, (j1, o1, d1) in enumerate(lookup[m]):
                for l2, (j2, o2, d2) in enumerate(lookup[m][l1 + 1:], l1 + 1):
                    x = model.addVar(vtype='B')
                    if use_big_m:
                        model.addCons(s[j2, o2] >= d1 + s[j1, o1] - bigM * (1 - x))
                        model.addCons(s[j1, o1] >= d2 + s[j2, o2] - bigM * x)
                    else:
                        model.addConsIndicator(s[j2, o2] - s[j1, o1] >= d1, binvar=x, activeone=True)
                        model.addConsIndicator(s[j1, o1] - s[j2, o2] >= d2, binvar=x, activeone=False)

        # model._instance = self
        # model._s = s

        if brancher is not None:
            # model.includeBranchrule(OverlapBrancher(s, tasks), "JSP Overlap Brancher",  # remainder, s, x, tasks
            #                         "Branch on the most overlapped tasks", priority=1000000, maxdepth=-1,
            #                         maxbounddist=1.0)
            pass

        model.setCharParam('lp/initalgorithm', 'd')
        model.setCharParam('lp/resolvealgorithm', 'd')
        model.setCharParam('estimation/restarts/restartpolicy', 'n')
        # model.setIntParam('display/freq', 600)
        # model.hideOutput(True)
        # model.setIntParam('branching/inference/priority', 11000)
        # model.setBoolParam('presolving/donotmultaggr', True)
        # model.setBoolParam('presolving/donotaggr', True)
        # model.setBoolParam('constraints/linear/aggregatevariables', False)
        # model.setBoolParam('constraints/setppc/cliqueshrinking', False)
        # model.setBoolParam('constraints/setppc/cliquelifting', True)
        # model.setIntParam('heuristics/conflictdiving/priority', 1000000)
        # model.setIntParam('heuristics/conflictdiving/freq', 3)
        # model.setIntParam('heuristics/alns/freq', 10)
        # model.setIntParam('heuristics/crossover/freq', 8)
        # model.setIntParam('heuristics/rens/freq', -1)
        # model.setIntParam('heuristics/rounding/freq', -1)
        # model.setIntParam('heuristics/shifting/freq', -1)
        # model.setIntParam('heuristics/rins/freq', 15)
        model.setIntParam('heuristics/conflictdiving/freq', -1)
        model.setIntParam('heuristics/rootsoldiving/freq', -1)
        model.setIntParam('heuristics/pscostdiving/freq', -1)
        model.setIntParam('heuristics/objpscostdiving/freq', -1)
        model.setIntParam('heuristics/linesearchdiving/freq', -1)
        model.setIntParam('heuristics/fracdiving/freq', -1)
        model.setIntParam('heuristics/guideddiving/freq', -1)
        # model.setIntParam('heuristics/alns/freq', -1)
        # model.setIntParam('constraints/setppc/sepafreq', 8)
        # model.setIntParam('separating/aggregation/freq', -1)
        # model.setCharParam('lp/pricing', 's')

        return model


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


# generator = jslg.BasicGenerator(
#     duration_range=(20, 200), seed=42, num_jobs=12, num_machines=12
# )

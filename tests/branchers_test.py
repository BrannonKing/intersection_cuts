import unittest

from .. import jsplib_loader as jl


class BranchersTestCases(unittest.TestCase):

    def test_basic_cplex_model(self):
        instances = jl.get_instances()
        instance = instances['abz5']
        model = instance.as_cplex_balas_model(True)
        model.solve()
        sln = model.solution
        print("SLN:", sln.get_status_string(), round(sln.get_objective_value()), instance.score)

    def test_basic_scip_model(self):
        instances = jl.get_instances()
        instance = instances['abz5']
        model = instance.as_scip_balas_model(True)
        model.optimize()
        model.printStatistics()

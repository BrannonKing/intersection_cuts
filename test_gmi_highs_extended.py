#!/usr/bin/env python3
"""Extended tests for HiGHS GMI cuts including bounds and example problems."""

import sys
sys.path.insert(0, '/home/brannon/Documents/Research/intersection_cuts')

import highspy as hp
import numpy as np
import highs_utils as hu
import example_loader
import gurobipy as gp


def test_knapsack_with_lower_bounds():
    """Test GMI cuts on knapsack with non-zero lower bounds."""
    print("=" * 60)
    print("Test: Knapsack with lower bounds")
    print("=" * 60)
    
    # Knapsack: maximize 5*x1 + 4*x2 + 3*x3
    # subject to: 2*x1 + 3*x2 + 2*x3 <= 10
    #             1 <= x1 <= 3, 0 <= x2 <= 2, 1 <= x3 <= 4
    
    h = hp.Highs()
    h.setOptionValue("output_flag", False)
    
    # Add variables with lower bounds (note: negate obj for maximization)
    h.addVariable(lb=1, ub=3, obj=-5, type=hp.HighsVarType.kContinuous)  # x1
    h.addVariable(lb=0, ub=2, obj=-4, type=hp.HighsVarType.kContinuous)  # x2
    h.addVariable(lb=1, ub=4, obj=-3, type=hp.HighsVarType.kContinuous)  # x3
    
    # Knapsack constraint
    h.addRow(-hp.kHighsInf, 10, 3,
             np.array([0, 1, 2], dtype=np.int32),
             np.array([2.0, 3.0, 2.0]))
    
    print("\nSolving LP relaxation...")
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    
    solution = h.getSolution()
    print(f"LP Solution: x1={solution.col_value[0]:.3f}, x2={solution.col_value[1]:.3f}, x3={solution.col_value[2]:.3f}")
    lp_obj = -h.getInfo().objective_function_value
    print(f"LP Objective: {lp_obj:.4f}")
    
    # Run GMI cuts
    print("\nRunning GMI cuts...")
    int_var_set = {0, 1, 2}
    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=5, verbose=True)
    
    print(f"\nResults:")
    print(f"  Starting objective: {-start_obj:.4f}")
    print(f"  Final objective: {-final_obj:.4f}")
    print(f"  Total cuts added: {num_cuts}")
    print(f"  Gap improvement: {-(start_obj - final_obj):.4f}")
    
    final_solution = h.getSolution()
    print(f"  Final solution: x1={final_solution.col_value[0]:.3f}, x2={final_solution.col_value[1]:.3f}, x3={final_solution.col_value[2]:.3f}")
    
    assert final_obj >= start_obj - 0.01, "Objective should improve"
    print("\n✓ Test passed!")


def test_knapsack_with_upper_bounds():
    """Test GMI cuts on knapsack with tight upper bounds."""
    print("\n" + "=" * 60)
    print("Test: Knapsack with upper bounds")
    print("=" * 60)
    
    # Knapsack: maximize 10*x1 + 8*x2 + 6*x3 + 4*x4
    # subject to: 5*x1 + 4*x2 + 3*x3 + 2*x4 <= 12
    #             0 <= x1 <= 2, 0 <= x2 <= 2, 0 <= x3 <= 3, 0 <= x4 <= 5
    
    h = hp.Highs()
    h.setOptionValue("output_flag", False)
    
    # Add variables with upper bounds
    h.addVariable(lb=0, ub=2, obj=-10, type=hp.HighsVarType.kContinuous)  # x1
    h.addVariable(lb=0, ub=2, obj=-8, type=hp.HighsVarType.kContinuous)   # x2
    h.addVariable(lb=0, ub=3, obj=-6, type=hp.HighsVarType.kContinuous)   # x3
    h.addVariable(lb=0, ub=5, obj=-4, type=hp.HighsVarType.kContinuous)   # x4
    
    # Knapsack constraint
    h.addRow(-hp.kHighsInf, 12, 4,
             np.array([0, 1, 2, 3], dtype=np.int32),
             np.array([5.0, 4.0, 3.0, 2.0]))
    
    print("\nSolving LP relaxation...")
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    
    solution = h.getSolution()
    print(f"LP Solution: {[f'{v:.3f}' for v in solution.col_value[:4]]}")
    lp_obj = -h.getInfo().objective_function_value
    print(f"LP Objective: {lp_obj:.4f}")
    
    # Run GMI cuts
    print("\nRunning GMI cuts...")
    int_var_set = {0, 1, 2, 3}
    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=5, verbose=True)
    
    print(f"\nResults:")
    print(f"  Starting objective: {-start_obj:.4f}")
    print(f"  Final objective: {-final_obj:.4f}")
    print(f"  Total cuts added: {num_cuts}")
    
    final_solution = h.getSolution()
    print(f"  Final solution: {[f'{v:.3f}' for v in final_solution.col_value[:4]]}")
    
    assert final_obj >= start_obj - 0.01, "Objective should improve"
    print("\n✓ Test passed!")


def test_gur_to_highs_converter():
    """Test the Gurobi to HiGHS model converter."""
    print("\n" + "=" * 60)
    print("Test: gur_to_highs converter")
    print("=" * 60)
    
    # Create a simple Gurobi model
    gur_model = gp.Model("test_convert")
    gur_model.params.OutputFlag = 0
    
    x = gur_model.addVar(name='x', vtype=gp.GRB.INTEGER, lb=0, ub=5)
    y = gur_model.addVar(name='y', vtype=gp.GRB.INTEGER, lb=1, ub=3)
    
    gur_model.setObjective(3*x + 2*y, sense=gp.GRB.MAXIMIZE)
    gur_model.addConstr(2*x + y <= 8, name="c1")
    gur_model.addConstr(x + 2*y <= 7, name="c2")
    gur_model.update()
    
    # Solve with Gurobi
    gur_model.optimize()
    gur_obj = gur_model.ObjVal
    gur_x = x.X
    gur_y = y.X
    print(f"Gurobi solution: x={gur_x:.3f}, y={gur_y:.3f}, obj={gur_obj:.3f}")
    
    # Convert to HiGHS (relaxed)
    highs_model = hu.gur_to_highs(gur_model, relaxed=True)
    
    # Solve with HiGHS
    status = highs_model.run()
    assert status == hp.HighsStatus.kOk
    
    highs_solution = highs_model.getSolution()
    # HiGHS reports the objective directly according to the sense set
    highs_obj = highs_model.getInfo().objective_function_value
    highs_x = highs_solution.col_value[0]
    highs_y = highs_solution.col_value[1]
    print(f"HiGHS solution: x={highs_x:.3f}, y={highs_y:.3f}, obj={highs_obj:.3f}")
    
    # Since Gurobi solves as IP and HiGHS as LP, objectives may differ
    # For maximization: HiGHS LP should be >= Gurobi IP
    # For minimization: HiGHS LP should be <= Gurobi IP
    if gur_model.ModelSense == gp.GRB.MAXIMIZE:
        assert highs_obj >= gur_obj - 0.01, f"HiGHS LP objective {highs_obj} should be >= Gurobi IP {gur_obj} for maximization"
    else:
        assert highs_obj <= gur_obj + 0.01, f"HiGHS LP objective {highs_obj} should be <= Gurobi IP {gur_obj} for minimization"
    
    print("\n✓ Converter test passed!")


def test_example_2d_below():
    """Test GMI cuts on 2D example from example_loader."""
    print("\n" + "=" * 60)
    print("Test: 2D example from below (example_loader)")
    print("=" * 60)
    
    instances = example_loader.get_instances()
    gur_model = instances["2Dbelow"].as_gurobi_model()
    
    print(f"Problem: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}, Constraints: {gur_model.NumConstrs}")
    
    # Convert to HiGHS (relaxed)
    h = hu.gur_to_highs(gur_model, relaxed=True)
    
    # Solve LP
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    
    solution = h.getSolution()
    print(f"LP Solution: x={solution.col_value[0]:.3f}, y={solution.col_value[1]:.3f}")
    lp_obj = -h.getInfo().objective_function_value  # Negate for maximization
    print(f"LP Objective: {lp_obj:.4f}")
    
    # Run GMI cuts
    assert gur_model.NumVars == gur_model.NumIntVars
    int_var_set = set(range(h.numVariables))
    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=5, verbose=True)
    
    print(f"\nResults:")
    print(f"  Starting objective: {-start_obj:.4f}")
    print(f"  Final objective: {-final_obj:.4f}")
    print(f"  Total cuts added: {num_cuts}")
    
    final_solution = h.getSolution()
    print(f"  Final solution: x={final_solution.col_value[0]:.3f}, y={final_solution.col_value[1]:.3f}")
    
    # Expected optimum should be 2
    expected_opt = instances["2Dbelow"].score
    print(f"  Expected optimum: {expected_opt}")
    
    print("\n✓ Test passed!")


def test_example_2d_second_round_preserves_optimum():
    """Regression: the second GMI round should keep the 2D optimum feasible."""

    instances = example_loader.get_instances()
    gur_model = instances["2Dbelow"].as_gurobi_model()

    h = hu.gur_to_highs(gur_model, relaxed=True)
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal

    expected_opt = instances["2Dbelow"].score
    objectives = []

    def capture_state(model: hp.Highs):
        objectives.append(model.getInfo().objective_function_value)

    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(
        h,
        int_var_set={0, 1},
        rounds=2,
        verbose=False,
        callback=capture_state,
    )

    # Callback fires before the first round and after each resolve
    assert len(objectives) >= 2, objectives

    solution = h.getSolution()
    x_val, y_val = solution.col_value[:2]

    assert abs(round(x_val) - x_val) <= 1e-6
    assert abs(round(y_val) - y_val) <= 1e-6
    assert round(y_val) == expected_opt

    assert final_obj >= expected_opt - 1e-6
    assert num_cuts >= 3  # first round produces two cuts, second at least one

    # Objective sequence should be non-increasing for maximization runs
    for earlier, later in zip(objectives, objectives[1:]):
        assert later <= earlier + 1e-8, objectives


def test_example_2d_ub():
    """Test GMI cuts on 2D example with upper bounded variable."""
    print("\n" + "=" * 60)
    print("Test: 2D with upper bounded x (example_loader)")
    print("=" * 60)
    
    instances = example_loader.get_instances()
    gur_model = instances["2DbelowUBx"].as_gurobi_model()
    
    print(f"Problem: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}, Constraints: {gur_model.NumConstrs}")
    
    # Show variable bounds
    for v in gur_model.getVars():
        print(f"  {v.VarName}: lb={v.LB}, ub={v.UB}")
    
    # Convert to HiGHS (relaxed)
    h = hu.gur_to_highs(gur_model, relaxed=True)
    
    # Solve LP
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    
    solution = h.getSolution()
    print(f"LP Solution: x={solution.col_value[0]:.3f}, y={solution.col_value[1]:.3f}")
    lp_obj = -h.getInfo().objective_function_value
    print(f"LP Objective: {lp_obj:.4f}")
    
    # Run GMI cuts
    int_var_set = set(range(h.numVariables))
    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=5, verbose=True)
    
    print(f"\nResults:")
    print(f"  Starting objective: {-start_obj:.4f}")
    print(f"  Final objective: {-final_obj:.4f}")
    print(f"  Total cuts added: {num_cuts}")
    
    final_solution = h.getSolution()
    print(f"  Final solution: x={final_solution.col_value[0]:.3f}, y={final_solution.col_value[1]:.3f}")
    
    expected_opt = instances["2DbelowUBx"].score
    print(f"  Expected optimum: {expected_opt}")
    assert round(final_obj) == expected_opt, "Final objective should match expected optimum"


def test_example_book_63():
    """Test GMI cuts on book example 6.3."""
    print("\n" + "=" * 60)
    print("Test: Book example 6.3 (3 variables)")
    print("=" * 60)
    
    instances = example_loader.get_instances()
    gur_model = instances["Book_6_3"].as_gurobi_model()
    
    print(f"Problem: {gur_model.ModelName}")
    print(f"Variables: {gur_model.NumVars}, Constraints: {gur_model.NumConstrs}")
    
    # Convert to HiGHS (relaxed)
    h = hu.gur_to_highs(gur_model, relaxed=True)
    
    # Solve LP
    status = h.run()
    assert status == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    
    solution = h.getSolution()
    print(f"LP Solution: {[f'{v:.3f}' for v in solution.col_value[:3]]}")
    lp_obj = -h.getInfo().objective_function_value
    print(f"LP Objective: {lp_obj:.4f}")
    
    # Run GMI cuts
    int_var_set = set(range(h.numVariables))
    start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=5, verbose=True)
    
    print(f"\nResults:")
    print(f"  Starting objective: {-start_obj:.4f}")
    print(f"  Final objective: {-final_obj:.4f}")
    print(f"  Total cuts added: {num_cuts}")
    
    final_solution = h.getSolution()
    print(f"  Final solution: {[f'{v:.3f}' for v in final_solution.col_value[:3]]}")
    
    expected_opt = instances["Book_6_3"].score
    print(f"  Expected optimum: {expected_opt}")
    assert final_obj >= expected_opt - 0.01, "Final objective should be at least the expected optimum"


def test_all_examples():
    """Test GMI cuts on all examples from example_loader."""
    print("\n" + "=" * 60)
    print("Test: All examples from example_loader")
    print("=" * 60)
    
    instances = example_loader.get_instances()
    failures = []
    
    for name, instance in instances.items():
        print(f"\n--- Testing {name} ---")
        gur_model = instance.as_gurobi_model()
        vtypes = gur_model.getAttr("VType")
        int_var_set = set(i for i, vt in enumerate(vtypes) if vt != gp.GRB.CONTINUOUS)
        
        try:
            # Convert to HiGHS (relaxed)
            h = hu.gur_to_highs(gur_model, relaxed=True)
            
            # Solve LP
            status = h.run()
            if status != hp.HighsStatus.kOk or h.getModelStatus() != hp.HighsModelStatus.kOptimal:
                print(f"  ⚠ LP solve failed: {h.getModelStatus()}")
                continue
            
            lp_obj = h.getInfo().objective_function_value
            print(f"  LP objective (HiGHS): {lp_obj:.4f}")
            print(f"  Problem sense: {'MAXIMIZE' if gur_model.ModelSense == gp.GRB.MAXIMIZE else 'MINIMIZE'}")
            
            # Run GMI cuts (2 rounds for speed)
            start_obj, final_obj, num_cuts = hu.run_gmi_cuts_highs(h, int_var_set, rounds=2, verbose=False)
            
            print(f"  After cuts (HiGHS): {final_obj:.4f} (cuts added: {num_cuts})")
            print(f"  Expected optimum: {instance.score}")
            
            # For maximization: LP gives upper bound, cuts tighten it down
            # After cuts, objective should be: expected_opt <= final_obj <= lp_obj
            # For minimization: LP gives lower bound, cuts tighten it up  
            # After cuts, objective should be: lp_obj <= final_obj <= expected_opt
            if gur_model.ModelSense == gp.GRB.MAXIMIZE:
                # Check that cuts didn't cut off the optimum (final_obj should be >= optimum)
                assert final_obj >= instance.score - 0.001, \
                    f"Cuts cut off optimum! Expected >= {instance.score}, got {final_obj:.6f}"
                # Check that cuts improved from LP (final_obj should be <= LP objective)
                assert final_obj <= start_obj + 0.001, \
                    f"Cuts made objective worse! LP was {start_obj:.6f}, after cuts {final_obj:.6f}"
            else:
                # Check that cuts didn't cut off the optimum (final_obj should be <= optimum)
                assert final_obj <= instance.score + 0.001, \
                    f"Cuts cut off optimum! Expected <= {instance.score}, got {final_obj:.6f}"
                # Check that cuts improved from LP (final_obj should be >= LP objective)
                assert final_obj >= start_obj - 0.001, \
                    f"Cuts made objective worse! LP was {start_obj:.6f}, after cuts {final_obj:.6f}"
            
            print(f"  ✓ Passed")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            failures.append(name + " - " + gur_model.ModelName)
    
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILED: {len(failures)} examples")
        for name in failures:
            print(f"  - {name}")
        print(f"{'='*60}")
    
    assert not failures, f"Failures in examples: {', '.join(failures)}"

def run_all_tests():
    """Run all extended tests."""
    tests = [
        test_knapsack_with_lower_bounds,
        test_knapsack_with_upper_bounds,
        test_gur_to_highs_converter,
        test_example_2d_below,
        test_example_2d_ub,
        test_example_book_63,
        test_all_examples,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if not failed:
        print("ALL EXTENDED TESTS PASSED!")
    else:
        print(f"FAILED TESTS: {', '.join(failed)}")
    print("=" * 60)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

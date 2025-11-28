# Tests Directory

All test files use relative imports to reference modules from the parent directory and follow pytest conventions.

## Test Organization

All test files are named `test_*.py` and contain test functions starting with `test_`:
- **test_*.py** - All tests follow pytest conventions with proper test functions and assertions
- Tests validate functionality by asserting expected behaviors rather than just printing output
- Optional dependencies (cplex, ntl_wrapper, cuppy) are handled with `pytest.importorskip()`

## Running Tests

### Run all tests with pytest (recommended):
```bash
cd /home/brannon/Documents/Research/intersection_cuts
python -m pytest tests/
```

### Run specific test file:
```bash
python -m pytest tests/test_nullspace_extension.py -v
```

### Run specific test function:
```bash
python -m pytest tests/test_seysen_reduce.py::test_seysen_reduce_trivial_identity -v
```

### Run with verbose output:
```bash
python -m pytest tests/ -v --tb=short
```

### Skip tests requiring optional dependencies:
Tests automatically skip if dependencies like `cplex`, `ntl_wrapper`, or `cuppy` are not available.

## Import Structure

All test files use relative imports:
```python
from .. import dikin_utils as du
from .. import gurobi_utils as gu
from .. import knapsack_loader as kl
```

This ensures tests can find the main codebase modules without sys.path manipulation.

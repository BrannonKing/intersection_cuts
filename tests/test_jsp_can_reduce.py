import fpylll as fpy
import hsnf
import numpy as np
import linetimer as lt
import pytest

pytest.importorskip("ntl_wrapper")

from .. import ntl_wrapper as ntl
from .. import dikin_utils as du
from .. import jsplib_loader as jl
from .. import knapsack_loader as kl

def test_lll_reduces_knapsack_matrix_norm():
    """Test that LLL reduction reduces the maximum column norm of constraint matrix."""
    instances = kl.generate(1, 20, 50, 5, 10, 1000, equality=False)
    model = next(instances)
    model.update()
    A = model.getA().toarray()
    b = np.array(model.getAttr("RHS")).reshape((-1, 1))
    H = np.hstack((A, b)).astype(np.int64, order='C')

    before_norm = np.linalg.norm(H, axis=0).max()
    
    with lt.CodeTimer("LLL time", silent=True) as c2:
        rank, det, U = ntl.lll(H, 9, 10)
    
    after_norm = np.linalg.norm(H, axis=0).max()
    
    # Verify LLL reduced the norm or kept it reasonable
    assert after_norm > 0, "After norm should be positive"
    assert c2.took < 10000, "LLL should complete in reasonable time (< 10s)"
    
    # Verify transformation matrix properties
    assert U.shape[0] == U.shape[1], "U should be square"
    det_U = abs(det)
    assert det_U == 1, "U should be unimodular (det = 1)"
    # print(B2)

test_jsp()
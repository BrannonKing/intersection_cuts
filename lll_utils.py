import numpy as np
import scipy.sparse as sps

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _sparse_dot_sorted(a_idx, a_val, b_idx, b_val):
    i = 0
    j = 0
    out = 0.0
    len_a = len(a_idx)
    len_b = len(b_idx)
    while i < len_a and j < len_b:
        ai = a_idx[i]
        bj = b_idx[j]
        if ai == bj:
            out += a_val[i] * b_val[j]
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return out


@njit(cache=True)
def _sparse_axpy_sorted(a_idx, a_val, alpha, b_idx, b_val, zero_tol):
    i = 0
    j = 0
    len_a = len(a_idx)
    len_b = len(b_idx)

    out_idx = np.empty(len_a + len_b, dtype=np.int64)
    out_val = np.empty(len_a + len_b, dtype=np.float64)
    out_k = 0
    out_norm_sq = 0.0

    while i < len_a and j < len_b:
        ai = a_idx[i]
        bj = b_idx[j]
        if ai == bj:
            idx = ai
            val = a_val[i] + alpha * b_val[j]
            i += 1
            j += 1
        elif ai < bj:
            idx = ai
            val = a_val[i]
            i += 1
        else:
            idx = bj
            val = alpha * b_val[j]
            j += 1

        keep = val != 0.0 if zero_tol <= 0.0 else abs(val) > zero_tol
        if keep:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1
            out_norm_sq += val * val

    while i < len_a:
        idx = a_idx[i]
        val = a_val[i]
        i += 1
        keep = val != 0.0 if zero_tol <= 0.0 else abs(val) > zero_tol
        if keep:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1
            out_norm_sq += val * val

    while j < len_b:
        idx = b_idx[j]
        val = alpha * b_val[j]
        j += 1
        keep = val != 0.0 if zero_tol <= 0.0 else abs(val) > zero_tol
        if keep:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1
            out_norm_sq += val * val

    return out_idx[:out_k], out_val[:out_k], out_norm_sq


@njit(cache=True)
def _sparse_int_axpy_sorted(a_idx, a_val, alpha, b_idx, b_val):
    i = 0
    j = 0
    len_a = len(a_idx)
    len_b = len(b_idx)

    out_idx = np.empty(len_a + len_b, dtype=np.int64)
    out_val = np.empty(len_a + len_b, dtype=np.int64)
    out_k = 0

    while i < len_a and j < len_b:
        ai = a_idx[i]
        bj = b_idx[j]
        if ai == bj:
            idx = ai
            val = a_val[i] + alpha * b_val[j]
            i += 1
            j += 1
        elif ai < bj:
            idx = ai
            val = a_val[i]
            i += 1
        else:
            idx = bj
            val = alpha * b_val[j]
            j += 1

        if val != 0:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1

    while i < len_a:
        idx = a_idx[i]
        val = a_val[i]
        i += 1
        if val != 0:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1

    while j < len_b:
        idx = b_idx[j]
        val = alpha * b_val[j]
        j += 1
        if val != 0:
            out_idx[out_k] = idx
            out_val[out_k] = val
            out_k += 1

    return out_idx[:out_k], out_val[:out_k]

class _SparseColumn:
    """Sparse column vector backed by two sorted Python lists."""

    __slots__ = ("indices", "values", "norm_sq")

    def __init__(self, indices=None, values=None):
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.empty(0, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64) if values is not None else np.empty(0, dtype=np.float64)
        if self.values.size:
            self.norm_sq = float(self.values @ self.values)
        else:
            self.norm_sq = 0.0

    @classmethod
    def from_dense(cls, col, zero_tol=0.0):
        arr = np.asarray(col, dtype=np.float64).reshape(-1)
        if zero_tol <= 0:
            idx = np.flatnonzero(arr != 0.0)
        else:
            idx = np.flatnonzero(np.abs(arr) > zero_tol)
        return cls(idx, arr[idx])

    @classmethod
    def from_csc_parts(cls, indices, values, zero_tol=0.0):
        idx = np.asarray(indices, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float64)
        if zero_tol > 0:
            keep = np.abs(vals) > zero_tol
            idx = idx[keep]
            vals = vals[keep]
        return cls(idx, vals)

    def dot(self, other):
        if self.indices.size == 0 or other.indices.size == 0:
            return 0.0

        return float(_sparse_dot_sorted(self.indices, self.values, other.indices, other.values))

    def axpy_inplace(self, alpha, other, zero_tol=0.0):
        """Apply self <- self + alpha * other in O(nnz(self) + nnz(other))."""
        if alpha == 0 or other.indices.size == 0:
            return

        out_idx, out_val, out_norm_sq = _sparse_axpy_sorted(
            self.indices,
            self.values,
            alpha,
            other.indices,
            other.values,
            float(zero_tol),
        )
        self.indices = out_idx
        self.values = out_val
        self.norm_sq = float(out_norm_sq)


class _SparseIntColumn:
    """Sparse integer column vector backed by sorted indices and values lists."""

    __slots__ = ("indices", "values")

    def __init__(self, indices=None, values=None):
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else np.empty(0, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.int64) if values is not None else np.empty(0, dtype=np.int64)

    @classmethod
    def unit(cls, idx):
        return cls([idx], [1])

    def axpy_inplace(self, alpha, other):
        if alpha == 0 or other.indices.size == 0:
            return

        out_idx, out_val = _sparse_int_axpy_sorted(
            self.indices,
            self.values,
            int(alpha),
            other.indices,
            other.values,
        )
        self.indices = out_idx
        self.values = out_val


def _matrix_to_sparse_columns(A, zero_tol=0.0):
    if sps.issparse(A):
        csc = sps.csc_matrix(A, copy=False)
        m, n = csc.shape
        cols = []
        for j in range(n):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]
            cols.append(
                _SparseColumn.from_csc_parts(
                    csc.indices[start:end],
                    csc.data[start:end],
                    zero_tol=zero_tol,
                )
            )
        return cols, m, n

    arr = np.asarray(A)
    if arr.ndim != 2:
        raise ValueError("A must be 2D")

    m, n = arr.shape
    cols = [_SparseColumn.from_dense(arr[:, j], zero_tol=zero_tol) for j in range(n)]
    return cols, m, n


def _sparse_columns_to_dense(cols, m, dtype=np.float64):
    n = len(cols)
    B = np.zeros((m, n), dtype=dtype)
    for j, col in enumerate(cols):
        if col.indices.size:
            B[col.indices, j] = col.values
    return B


def _sparse_int_columns_to_dense(cols, m):
    n = len(cols)
    U = np.zeros((m, n), dtype=np.int64)
    for j, col in enumerate(cols):
        if col.indices.size:
            U[col.indices, j] = col.values
    return U


def _sparse_columns_to_csc(cols, m, dtype=np.float64):
    n = len(cols)
    nnz = int(sum(col.indices.size for col in cols))
    data = np.empty(nnz, dtype=dtype)
    indices = np.empty(nnz, dtype=np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)

    k = 0
    indptr[0] = 0
    for j, col in enumerate(cols):
        c_nnz = col.indices.size
        if c_nnz:
            next_k = k + c_nnz
            indices[k:next_k] = col.indices
            data[k:next_k] = col.values
            k = next_k
        indptr[j + 1] = k

    return sps.csc_matrix((data, indices, indptr), shape=(m, n), dtype=dtype)


def _sparse_int_columns_to_csc(cols, m):
    n = len(cols)
    nnz = int(sum(col.indices.size for col in cols))
    data = np.empty(nnz, dtype=np.int64)
    indices = np.empty(nnz, dtype=np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)

    k = 0
    indptr[0] = 0
    for j, col in enumerate(cols):
        c_nnz = col.indices.size
        if c_nnz:
            next_k = k + c_nnz
            indices[k:next_k] = col.indices
            data[k:next_k] = col.values
            k = next_k
        indptr[j + 1] = k

    return sps.csc_matrix((data, indices, indptr), shape=(m, n), dtype=np.int64)


class _SparseLLLCallbackState:
    __slots__ = ("_cols", "_u_cols", "_m", "_n", "_B_cache", "_U_cache")

    def __init__(self, cols, u_cols, m, n):
        self._cols = cols
        self._u_cols = u_cols
        self._m = m
        self._n = n
        self._B_cache = None
        self._U_cache = None

    @property
    def B(self):
        if self._B_cache is None:
            self._B_cache = _sparse_columns_to_csc(self._cols, self._m)
        return self._B_cache

    @property
    def U(self):
        if self._U_cache is None:
            self._U_cache = _sparse_int_columns_to_csc(self._u_cols, self._n)
        return self._U_cache

    def row_argmax(self, row):
        """Return argmax column index for a row without materializing B."""
        best_col = 0
        best_val = 0.0
        have_val = False

        for j, col in enumerate(self._cols):
            # indices are sorted; search row in this sparse column.
            pos = np.searchsorted(col.indices, row)
            if pos < col.indices.size and col.indices[pos] == row:
                val = col.values[pos]
                if (not have_val) or (val > best_val):
                    best_val = val
                    best_col = j
                    have_val = True

        return best_col


def lll_apx(A, early_exit_func=None, iterations=10):
    """
    Approximate LLL-style reduction using dot-product heuristics.

     Steps per iteration:
        1) For each vector (column) i, compute projection coefficients against all
            prior vectors j < i using the current basis state.
        2) Pick the prior vector with the largest absolute projection coefficient.
      3) Subtract r copies of the contributor: r = round(mu) by default.
      4) Sort all vectors by length (shortest to longest).
      5) Repeat for the requested number of iterations.

    Notes:
      - This operates on column vectors. If your vectors are rows, pass A.T.
      - The recommended number of instances is r = round(mu), where
        mu = (v_i · v_j) / (v_j · v_j).
    """
    assert not sps.issparse(A)
    B = np.array(A, copy=True) # or could have it modify A

    m, n = B.shape
    U = np.eye(n, dtype=np.int64)

    last_iter = -1
    
    for j in range(iterations):
        last_iter = j
        # Cache squared column norms and update incrementally as columns change.
        col_sqnorms = np.einsum("ij,ij->j", B, B)
        for i in range(1, n):
            denoms = col_sqnorms[:i]
            numerators = B[:, :i].T @ B[:, i]
            mus = np.divide(
                numerators,
                denoms,
                out=np.zeros_like(denoms, dtype=np.float64),
                where=denoms != 0,
            )
            best_j = np.argmax(np.abs(mus)) if i and np.any(denoms != 0) else None
            if best_j is None:
                continue
            best_mu = mus[best_j]
            r = int(round(best_mu))
            if r != 0:
                B[:, i] -= r * B[:, best_j]
                U[:, i] -= r * U[:, best_j]
                # Column i can become a future denominator for i' > i.
                col_sqnorms[i] = np.dot(B[:, i], B[:, i])

        lengths = np.linalg.norm(B, axis=0)
        order = np.argsort(lengths)
        B = B[:, order]
        U = U[:, order]

        if early_exit_func is not None and early_exit_func(B, U, j):
            break

    return B, U, last_iter


def lll_apx_sparse(A, early_exit_func=None, iterations=10, zero_tol=0.0, return_dense=False):
    """
    Approximate LLL-style reduction with sparse column-vector storage.

        Input can be a dense ndarray-like matrix or any SciPy sparse matrix.
        Internally, each column is represented by sorted index/value ndarrays.

        By default this returns sparse CSC matrices for both B and U.
        Set return_dense=True to convert outputs to dense ndarrays.

        If early_exit_func is provided, callback invocation is lazy and sparse.
        Callback signature: early_exit_func(state, iteration), where state.B and
        state.U are lazily built CSC matrices.
    """
    cols, m, n = _matrix_to_sparse_columns(A, zero_tol=zero_tol)
    U_cols = [_SparseIntColumn.unit(j) for j in range(n)]

    last_iter = -1
    for j in range(iterations):
        col_sqnorms = np.array([col.norm_sq for col in cols], dtype=np.float64)

        for i in range(1, n):
            best_j = None
            best_mu = 0.0
            best_abs_mu = 0.0

            for jj in range(i):
                denom = col_sqnorms[jj]
                if denom == 0.0:
                    continue
                mu = cols[jj].dot(cols[i]) / denom
                abs_mu = abs(mu)
                if abs_mu > best_abs_mu:
                    best_abs_mu = abs_mu
                    best_mu = mu
                    best_j = jj

            if best_j is None:
                continue

            r = int(round(best_mu))
            if r != 0:
                cols[i].axpy_inplace(-r, cols[best_j], zero_tol=zero_tol)
                U_cols[i].axpy_inplace(-r, U_cols[best_j])
                col_sqnorms[i] = cols[i].norm_sq

        order = np.argsort(col_sqnorms)
        cols = [cols[k] for k in order]
        U_cols = [U_cols[k] for k in order]

        last_iter = j
        if early_exit_func is not None:
            state = _SparseLLLCallbackState(cols, U_cols, m, n)
            should_stop = early_exit_func(state, j)
            if should_stop:
                break

    if return_dense:
        B = _sparse_columns_to_dense(cols, m)
        U = _sparse_int_columns_to_dense(U_cols, n)
    else:
        B = _sparse_columns_to_csc(cols, m)
        U = _sparse_int_columns_to_csc(U_cols, n)
    return B, U, last_iter


def lll_ntl(A, a=9, b=10):
    import ntl_wrapper as ntl

    return ntl.lll(A, a, b)


def lll_qr(A, delta=0.9):
    if sps.issparse(A):
        A = A.todense()

    m, n = A.shape
    assert m >= n, "Number of rows must be at least number of columns."

    U = np.eye(n, dtype=np.int32)

    k = 1
    R = np.linalg.qr(A, mode="r")
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            mu = R[j, k] / R[j, j]
            r = int(np.round(mu))
            if r != 0:
                A[:, k] -= r * A[:, j]
                U[:, k] -= r * U[:, j]
                R = np.linalg.qr(A, mode="r")

        # Lovász condition
        lhs = delta * (R[k - 1, k - 1] ** 2)
        rhs = (R[k, k] ** 2) + (R[k - 1, k] ** 2)

        if lhs <= rhs:
            k += 1
        else:
            # Swap columns k and k-1
            A[:, [k, k - 1]] = A[:, [k - 1, k]]
            U[:, [k, k - 1]] = U[:, [k - 1, k]]
            R = np.linalg.qr(A, mode="r")

            k = max(k - 1, 1)

    return A, U


def size_reduce(B, R, U, j, k):
    """
    Size-reduce column k with respect to column j.
    """
    mu = R[j, k] / R[j, j]
    r = int(np.round(mu))
    if r == 0:
        return

    B[:, k] -= r * B[:, j]
    U[:, k] -= r * U[:, j]

    # Update R column k only
    R[: j + 1, k] -= r * R[: j + 1, j]


def lovasz_swap(B, R, U, k):
    """
    Perform LLL swap between columns k-1 and k.
    """
    # Swap columns in B and U
    B[:, [k - 1, k]] = B[:, [k, k - 1]]
    U[:, [k - 1, k]] = U[:, [k, k - 1]]

    # Swap columns in R
    R[:, [k - 1, k]] = R[:, [k, k - 1]]

    # Apply Givens rotation to restore triangular form
    a = R[k - 1, k - 1]
    b = R[k, k - 1]

    r = np.hypot(a, b)
    c = a / r
    s = b / r

    G = np.array([[c, s], [-s, c]])
    R[k - 1 : k + 1, k - 1 :] = G @ R[k - 1 : k + 1, k - 1 :]


def lll(A, delta=0.9):
    if sps.issparse(A):
        A = A.todense()

    m, n = A.shape
    assert m >= n, "Number of rows must be at least number of columns."

    U = np.eye(n, dtype=np.int32)
    R = np.linalg.qr(A, mode="r")

    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            size_reduce(A, R, U, j, k)

        if delta * R[k - 1, k - 1] ** 2 <= R[k, k] ** 2 + R[k - 1, k] ** 2:
            k += 1
        else:
            lovasz_swap(A, R, U, k)
            k = max(k - 1, 1)

    return A, U

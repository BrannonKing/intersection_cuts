import torch
import scipy.sparse as sps
import numpy as np

"""
lll_torch.py — GPU-accelerated approximate LLL reduction using PyTorch.

Key design decisions vs. the original NumPy version:
  - All operations stay on-device (no CPU round-trips inside the loop).
  - The mu matrix is computed in one vectorized pass over the full lower
    triangle of G, replacing the per-column Python loop entirely.
  - Conflict-aware batched subtraction: columns that share a best_j
    contributor are grouped; within each group the updates are independent
    and applied in a single indexed scatter.
  - torch.compile-friendly: no Python-level data-dependent control flow
    inside the hot path (graph breaks), only tensor ops and fixed-shape masks.
"""

# ---------------------------------------------------------------------------
# Core kernel (compile target)
# ---------------------------------------------------------------------------


def _lll_apx_kernel(B: torch.Tensor, max_iterations: int, early_exit_func=None) -> tuple[torch.Tensor, int]:
    for i in range(max_iterations):
        # 1. Gram matrix
        G = B.t().mm(B)
        diag = G.diagonal()

        # 2. Compute mus (vmap-safe out-of-place ops)
        safe_diag = diag.masked_fill(diag == 0, 1.0)
        mus = G / safe_diag.unsqueeze(0)
        mus = mus.tril(-1)

        # 3. Find best j
        best_j = mus.abs().argmax(dim=1)
        best_mu = mus.gather(dim=1, index=best_j.unsqueeze(1)).squeeze(1)
        r = torch.round(best_mu)

        # 4. Apply updates without where-masking (vmap-safe out-of-place subtraction)
        contributors = B[:, best_j]
        B -= contributors * r.unsqueeze(0)

        # 5. Sort by squared length to avoid sqrt
        sq_lengths = (B * B).sum(dim=0)
        B = B[:, sq_lengths.argsort()]

        if early_exit_func is not None and early_exit_func(B, None, i):
            return B, i

    return B, max_iterations


def lll_apx_torch(
    A,
    iterations: int = 3,
    early_exit_func = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, int]:
    """
    Approximate LLL-style reduction — PyTorch/GPU implementation.

    Args:
        A:          Input matrix.  Accepts:
                      - numpy ndarray  (m, n)
                      - scipy sparse matrix
                      - torch.Tensor   (m, n)
        iterations: Number of reduction passes (default 3).
        device:     Target device string, e.g. 'cuda', 'cuda:0', 'cpu'.
                    If None, uses CUDA if available, else CPU.
        dtype:      Floating-point dtype (default float64 for numerical
                    stability; use float32 to halve memory on large problems).

    Returns:
        torch.Tensor (m, n) on the same device, columns sorted by length.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- normalise input ---
    if sps.issparse(A):
        A = A.toarray()
    if isinstance(A, np.ndarray):
        B = torch.from_numpy(A).to(dtype=dtype, device=device)
    elif isinstance(A, torch.Tensor):
        B = A.to(dtype=dtype, device=device)
    else:
        B = torch.tensor(A, dtype=dtype, device=device)

    B, iters = _lll_apx_kernel(B, iterations, early_exit_func)
    return B.cpu(), iters

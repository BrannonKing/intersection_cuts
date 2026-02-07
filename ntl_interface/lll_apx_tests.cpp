#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "lll_utils.hpp"

class LllApxTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};

    Eigen::MatrixXd randomIntMatrix(int rows, int cols, int low = -10, int high = 10) {
        std::uniform_int_distribution<int> dist(low, high);
        Eigen::MatrixXd M(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                M(i, j) = dist(rng);
        return M;
    }

    // Ensure the matrix has full row rank by regenerating if needed
    Eigen::MatrixXd randomFullRowRankMatrix(int rows, int cols, int low = -10, int high = 10) {
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto M = randomIntMatrix(rows, cols, low, high);
            Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
            if (lu.rank() == rows)
                return M;
        }
        // Fallback: construct one deterministically
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(rows, cols);
        for (int i = 0; i < rows; ++i)
            M(i, i) = 1.0;
        return M;
    }

    SparseMatrixXd denseToSparse(const Eigen::MatrixXd& D) {
        SparseMatrixXd S(D.rows(), D.cols());
        std::vector<Eigen::Triplet<double>> triplets;
        for (Eigen::Index i = 0; i < D.rows(); ++i)
            for (Eigen::Index j = 0; j < D.cols(); ++j)
                if (D(i, j) != 0.0)
                    triplets.emplace_back(i, j, D(i, j));
        S.setFromTriplets(triplets.begin(), triplets.end());
        return S;
    }

    Eigen::MatrixXd sparseToDense(const SparseMatrixXd& S) {
        return Eigen::MatrixXd(S);
    }
};

// ---------------------------------------------------------------------------
// Dense lll_apx: lattice volume (Gram determinant) is preserved
// Only meaningful when B has full column rank (m >= n and rank == n),
// otherwise det(B^T B) is 0 and floating-point noise dominates.
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DensePreservesLatticeVolume) {
    int tested = 0;
    for (int trial = 0; trial < 60 && tested < 25; ++trial) {
        int n = 2 + (rng() % 5); // cols in [2,6]
        int m = n + (rng() % 3); // rows >= cols, in [n, n+2], so full col rank is possible
        if (m > 6) m = 6;
        if (m < n) continue;

        auto B = randomIntMatrix(m, n);

        // Only test when B has full column rank
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (lu.rank() != n) continue;

        auto B_orig = B;

        lll_apx(B, 50);

        // Gram determinant = det(B^T B) should be unchanged by unimodular col ops
        Eigen::MatrixXd gram_orig = B_orig.transpose() * B_orig;
        Eigen::MatrixXd gram_new  = B.transpose() * B;

        double det_orig = gram_orig.determinant();
        double det_new  = gram_new.determinant();

        double tol = std::max(std::abs(det_orig) * 1e-6, 1e-8);
        EXPECT_NEAR(det_orig, det_new, tol)
            << "Gram determinant changed for " << m << "x" << n
            << " matrix, trial " << trial;
        ++tested;
    }
    EXPECT_GE(tested, 10) << "Too few full-column-rank cases generated";
}

// ---------------------------------------------------------------------------
// Dense lll_apx: rank of the matrix is preserved
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DensePreservesRank) {
    for (int trial = 0; trial < 30; ++trial) {
        int m = 2 + (rng() % 5);
        int n = 2 + (rng() % 5);

        auto B = randomIntMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_orig(B);
        auto rank_orig = lu_orig.rank();

        lll_apx(B, 50);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_new(B);
        auto rank_new = lu_new.rank();

        EXPECT_EQ(rank_orig, rank_new)
            << "Rank changed from " << rank_orig << " to " << rank_new
            << " for " << m << "x" << n << " matrix, trial " << trial;
    }
}

// ---------------------------------------------------------------------------
// Dense lll_apx: null-space property is preserved
//
// If N = kernel(A), then A*N = 0.  After lll_apx(N) the columns are integer
// linear combinations of the original columns, so A*N_reduced must still ≈ 0.
// We also check that the reduced basis still spans the full null space.
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DensePreservesNullSpace) {
    int tested = 0;
    for (int trial = 0; trial < 50 && tested < 25; ++trial) {
        int m = 1 + (rng() % 4);           // rows in [1,4]
        int n = m + 1 + (rng() % 3);       // cols > rows, at most m+3
        if (n > 6) n = 6;

        auto A = randomFullRowRankMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
        Eigen::MatrixXd N = lu.kernel();
        if (N.cols() == 0) continue;

        // Sanity: A * N should already be ≈ 0
        double pre_residual = (A * N).norm();
        ASSERT_LT(pre_residual, 1e-10)
            << "Eigen kernel computation itself is wrong (trial " << trial << ")";

        Eigen::Index null_dim = N.cols();
        Eigen::MatrixXd N_orig = N;

        lll_apx(N, 50);

        // 1) Null-space property: every column of the reduced basis is in ker(A)
        Eigen::MatrixXd residual = A * N;
        double residual_norm = residual.norm();
        EXPECT_LT(residual_norm, 1e-8)
            << "Null-space property violated after lll_apx for "
            << m << "x" << n << " matrix (null dim " << null_dim
            << "), residual norm = " << residual_norm
            << ", trial " << trial;

        // 2) The reduced basis must still span the full null space (rank preserved)
        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N);
        EXPECT_EQ(lu_N.rank(), null_dim)
            << "Null-space basis rank changed from " << null_dim
            << " to " << lu_N.rank() << " (trial " << trial << ")";

        ++tested;
    }
    EXPECT_GE(tested, 10) << "Too few non-trivial null-space cases were generated";
}

// ---------------------------------------------------------------------------
// Dense lll_apx: columns are not longer than the original longest column
// (a very weak sanity check that reduction is doing *something*)
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DenseReducesOrMaintainsNorms) {
    for (int trial = 0; trial < 20; ++trial) {
        int m = 3 + (rng() % 4); // rows in [3,6]
        int n = 2 + (rng() % 5); // cols in [2,6]

        auto B = randomIntMatrix(m, n, -20, 20);
        auto B_orig = B;

        lll_apx(B, 100);

        double max_orig_sq = 0;
        for (Eigen::Index j = 0; j < B_orig.cols(); ++j)
            max_orig_sq = std::max(max_orig_sq, B_orig.col(j).squaredNorm());

        // The shortest vector in the reduced basis should not exceed
        // the longest vector in the original basis (very loose bound).
        double min_reduced_sq = std::numeric_limits<double>::max();
        for (Eigen::Index j = 0; j < B.cols(); ++j)
            min_reduced_sq = std::min(min_reduced_sq, B.col(j).squaredNorm());

        EXPECT_LE(min_reduced_sq, max_orig_sq + 1e-10)
            << "Shortest reduced vector is longer than longest original vector"
            << " (trial " << trial << ")";
    }
}

// ---------------------------------------------------------------------------
// Sparse lll_apx_sparse: lattice volume (Gram determinant) is preserved
// Only meaningful when B has full column rank (m >= n and rank == n).
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, SparsePreservesLatticeVolume) {
    int tested = 0;
    for (int trial = 0; trial < 60 && tested < 25; ++trial) {
        int n = 2 + (rng() % 5);
        int m = n + (rng() % 3);
        if (m > 6) m = 6;
        if (m < n) continue;

        auto D = randomIntMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(D);
        if (lu.rank() != n) continue;

        auto D_orig = D;

        SparseMatrixXd S = denseToSparse(D);
        SparseMatrixXd S_result = lll_apx_sparse(S, 50);
        Eigen::MatrixXd B_result = sparseToDense(S_result);

        Eigen::MatrixXd gram_orig = D_orig.transpose() * D_orig;
        Eigen::MatrixXd gram_new  = B_result.transpose() * B_result;

        double det_orig = gram_orig.determinant();
        double det_new  = gram_new.determinant();

        double tol = std::max(std::abs(det_orig) * 1e-6, 1e-8);
        EXPECT_NEAR(det_orig, det_new, tol)
            << "Sparse: Gram determinant changed for " << m << "x" << n
            << " matrix, trial " << trial;
        ++tested;
    }
    EXPECT_GE(tested, 10) << "Too few full-column-rank cases generated";
}

// ---------------------------------------------------------------------------
// Sparse lll_apx_sparse: rank of the matrix is preserved
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, SparsePreservesRank) {
    for (int trial = 0; trial < 30; ++trial) {
        int m = 2 + (rng() % 5);
        int n = 2 + (rng() % 5);

        auto D = randomIntMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_orig(D);
        auto rank_orig = lu_orig.rank();

        SparseMatrixXd S = denseToSparse(D);
        SparseMatrixXd S_result = lll_apx_sparse(S, 50);
        Eigen::MatrixXd D_result = sparseToDense(S_result);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_new(D_result);
        auto rank_new = lu_new.rank();

        EXPECT_EQ(rank_orig, rank_new)
            << "Sparse: Rank changed from " << rank_orig << " to " << rank_new
            << " for " << m << "x" << n << " matrix, trial " << trial;
    }
}

// ---------------------------------------------------------------------------
// Sparse lll_apx_sparse: null-space property is preserved
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, SparsePreservesNullSpace) {
    int tested = 0;
    for (int trial = 0; trial < 50 && tested < 25; ++trial) {
        int m = 1 + (rng() % 4);
        int n = m + 1 + (rng() % 3);
        if (n > 6) n = 6;

        auto A = randomFullRowRankMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
        Eigen::MatrixXd N = lu.kernel();
        if (N.cols() == 0) continue;

        double pre_residual = (A * N).norm();
        ASSERT_LT(pre_residual, 1e-10)
            << "Eigen kernel computation itself is wrong (trial " << trial << ")";

        Eigen::Index null_dim = N.cols();

        SparseMatrixXd N_sparse = denseToSparse(N);
        SparseMatrixXd N_reduced_sparse = lll_apx_sparse(N_sparse, 50);
        Eigen::MatrixXd N_reduced = sparseToDense(N_reduced_sparse);

        // 1) Every column of the reduced basis is in ker(A)
        Eigen::MatrixXd residual = A * N_reduced;
        double residual_norm = residual.norm();
        EXPECT_LT(residual_norm, 1e-8)
            << "Sparse: null-space property violated for "
            << m << "x" << n << " matrix (null dim " << null_dim
            << "), residual norm = " << residual_norm
            << ", trial " << trial;

        // 2) Rank of null-space basis preserved
        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N_reduced);
        EXPECT_EQ(lu_N.rank(), null_dim)
            << "Sparse: null-space basis rank changed from " << null_dim
            << " to " << lu_N.rank() << " (trial " << trial << ")";

        ++tested;
    }
    EXPECT_GE(tested, 10) << "Too few non-trivial null-space cases were generated";
}

// ---------------------------------------------------------------------------
// Dense & Sparse both produce valid null-space bases on the same input.
// Note: the two implementations differ in inner-product staleness (dense
// pre-computes the Gram matrix; sparse evaluates dots on the fly), so they
// may not yield identical output — but both must preserve the null-space
// property and basis rank.
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DenseAndSparseBothValidNullSpace) {
    int tested = 0;
    for (int trial = 0; trial < 60 && tested < 20; ++trial) {
        int m = 1 + (rng() % 4);
        int n = m + 1 + (rng() % 3);
        if (n > 6) n = 6;
        if (n <= m) continue;

        auto A = randomFullRowRankMatrix(m, n);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
        Eigen::MatrixXd N = lu.kernel();
        if (N.cols() == 0) continue;
        Eigen::Index null_dim = N.cols();

        // Dense path
        Eigen::MatrixXd N_dense = N;
        lll_apx(N_dense, 50);

        // Sparse path
        SparseMatrixXd N_sparse = denseToSparse(N);
        SparseMatrixXd N_sparse_result = lll_apx_sparse(N_sparse, 50);
        Eigen::MatrixXd N_from_sparse = sparseToDense(N_sparse_result);

        // Both must satisfy A * N_reduced ≈ 0
        double res_dense  = (A * N_dense).norm();
        double res_sparse = (A * N_from_sparse).norm();
        EXPECT_LT(res_dense, 1e-8)
            << "Dense null-space violated for " << m << "x" << n
            << " matrix, trial " << trial;
        EXPECT_LT(res_sparse, 1e-8)
            << "Sparse null-space violated for " << m << "x" << n
            << " matrix, trial " << trial;

        // Both must preserve null-space rank
        Eigen::FullPivLU<Eigen::MatrixXd> lu_d(N_dense);
        Eigen::FullPivLU<Eigen::MatrixXd> lu_s(N_from_sparse);
        EXPECT_EQ(lu_d.rank(), null_dim)
            << "Dense rank changed, trial " << trial;
        EXPECT_EQ(lu_s.rank(), null_dim)
            << "Sparse rank changed, trial " << trial;

        ++tested;
    }
    EXPECT_GE(tested, 10) << "Too few valid test cases generated";
}

// ---------------------------------------------------------------------------
// Dense lll_apx: null-space with integer-constructed inputs
// Construct A and N so that A*N = 0 exactly with integer entries.
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DenseIntegerNullSpaceBasis) {
    // Test case 1: A = [1 2 3; 4 5 6], null space = span{[1, -2, 1]^T}
    {
        Eigen::MatrixXd A(2, 3);
        A << 1, 2, 3,
             4, 5, 6;

        Eigen::MatrixXd N(3, 1);
        N << 1, -2, 1;

        ASSERT_LT((A * N).norm(), 1e-12);

        lll_apx(N, 50);

        double residual = (A * N).norm();
        EXPECT_LT(residual, 1e-10)
            << "Integer null-space violated, residual = " << residual;
    }

    // Test case 2: known 2D null space
    {
        // A = [1 0 1 0; 0 1 0 1] => null space = span{[-1,0,1,0], [0,-1,0,1]}
        Eigen::MatrixXd A(2, 4);
        A << 1, 0, 1, 0,
             0, 1, 0, 1;

        Eigen::MatrixXd N(4, 2);
        N << -1,  0,
              0, -1,
              1,  0,
              0,  1;

        ASSERT_LT((A * N).norm(), 1e-12);

        lll_apx(N, 50);

        double residual = (A * N).norm();
        EXPECT_LT(residual, 1e-10)
            << "Integer 2D null-space violated, residual = " << residual;

        Eigen::FullPivLU<Eigen::MatrixXd> lu(N);
        EXPECT_EQ(lu.rank(), 2) << "Null-space rank should remain 2";
    }

    // Test case 3: randomly generated integer null-space matrices
    // Build A from a known null space: pick random integer N, then build A
    // as the left-null-space of N^T (equivalently, rows of A are orthogonal to cols of N).
    for (int trial = 0; trial < 15; ++trial) {
        int n = 3 + (rng() % 4);           // ambient dim in [3,6]
        int k = 1 + (rng() % (n - 1));     // null-space dim in [1, n-1]
        int m = n - k;                       // rows of A

        // Build random integer null-space basis
        Eigen::MatrixXd N = randomIntMatrix(n, k, -5, 5);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N);
        if (lu_N.rank() != k) continue; // need full-rank N

        // Build A whose rows are orthogonal to columns of N.
        // Use the orthogonal complement of N's column space.
        // QR decomposition of N gives us Q = [Q1 | Q2], and A = Q2^T works.
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(N);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
        // Q2 = last (n - k) columns of Q
        Eigen::MatrixXd A = Q.rightCols(m).transpose();

        // Verify A * N ≈ 0
        double pre_check = (A * N).norm();
        ASSERT_LT(pre_check, 1e-8)
            << "Construction of A failed (trial " << trial << ")";

        Eigen::MatrixXd N_copy = N;
        lll_apx(N_copy, 80);

        // Null-space property
        double residual = (A * N_copy).norm();
        EXPECT_LT(residual, 1e-7)
            << "Integer null-space violated after lll_apx, "
            << n << "-dim, null dim " << k
            << ", residual = " << residual
            << " (trial " << trial << ")";

        // Rank preserved
        Eigen::FullPivLU<Eigen::MatrixXd> lu_reduced(N_copy);
        EXPECT_EQ(lu_reduced.rank(), k)
            << "Null-space rank changed after lll_apx (trial " << trial << ")";
    }
}

// ---------------------------------------------------------------------------
// Sparse lll_apx_sparse: integer null-space tests (mirrors the dense ones)
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, SparseIntegerNullSpaceBasis) {
    for (int trial = 0; trial < 15; ++trial) {
        int n = 3 + (rng() % 4);
        int k = 1 + (rng() % (n - 1));
        int m = n - k;

        Eigen::MatrixXd N = randomIntMatrix(n, k, -5, 5);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N);
        if (lu_N.rank() != k) continue;

        Eigen::HouseholderQR<Eigen::MatrixXd> qr(N);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd A = Q.rightCols(m).transpose();

        double pre_check = (A * N).norm();
        ASSERT_LT(pre_check, 1e-8);

        SparseMatrixXd N_sparse = denseToSparse(N);
        SparseMatrixXd N_reduced = lll_apx_sparse(N_sparse, 80);
        Eigen::MatrixXd N_dense_result = sparseToDense(N_reduced);

        double residual = (A * N_dense_result).norm();
        EXPECT_LT(residual, 1e-7)
            << "Sparse: integer null-space violated, "
            << n << "-dim, null dim " << k
            << ", residual = " << residual
            << " (trial " << trial << ")";

        Eigen::FullPivLU<Eigen::MatrixXd> lu_reduced(N_dense_result);
        EXPECT_EQ(lu_reduced.rank(), k)
            << "Sparse: null-space rank changed (trial " << trial << ")";
    }
}

// ---------------------------------------------------------------------------
// Edge case: single-column matrix — should be returned unchanged
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, SingleColumnUnchanged) {
    Eigen::MatrixXd B(4, 1);
    B << 3, -1, 4, 2;
    Eigen::MatrixXd B_orig = B;

    lll_apx(B, 50);

    EXPECT_LT((B - B_orig).norm(), 1e-14)
        << "Single-column matrix was modified by lll_apx";

    SparseMatrixXd S = denseToSparse(B_orig);
    SparseMatrixXd S_result = lll_apx_sparse(S, 50);
    Eigen::MatrixXd S_dense = sparseToDense(S_result);

    EXPECT_LT((S_dense - B_orig).norm(), 1e-14)
        << "Single-column matrix was modified by lll_apx_sparse";
}

// ---------------------------------------------------------------------------
// Edge case: identity matrix — already reduced, volume = 1
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, IdentityMatrixPreserved) {
    for (int n = 2; n <= 6; ++n) {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd I_copy = I;

        lll_apx(I_copy, 50);

        // Columns should be permutations of ±e_i, so Gram = I
        Eigen::MatrixXd gram = I_copy.transpose() * I_copy;
        EXPECT_LT((gram - Eigen::MatrixXd::Identity(n, n)).norm(), 1e-12)
            << "Identity " << n << "x" << n << " not preserved";
    }
}

// ---------------------------------------------------------------------------
// Callback version: check that the callback fires and can stop early
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DenseCallbackStopsEarly) {
    Eigen::MatrixXd B = randomIntMatrix(5, 4, -20, 20);

    int callback_count = 0;
    lll_apx(B, 100, [&](const int it, const Eigen::MatrixXd& M) -> bool {
        ++callback_count;
        // Stop after 3 iterations
        return callback_count >= 3;
    });

    EXPECT_EQ(callback_count, 3)
        << "Callback should have been called exactly 3 times";
}

TEST_F(LllApxTest, SparseCallbackStopsEarly) {
    Eigen::MatrixXd D = randomIntMatrix(5, 4, -20, 20);
    SparseMatrixXd S = denseToSparse(D);

    int callback_count = 0;
    auto result = lll_apx_sparse(S, 100, [&](const int it, const SparseMatrixXd& M) -> bool {
        ++callback_count;
        return callback_count >= 3;
    });

    EXPECT_EQ(callback_count, 3)
        << "Sparse callback should have been called exactly 3 times";
}

// ---------------------------------------------------------------------------
// Stress test: many random matrices, all under 7 dimensions, checking all
// key invariants together.
// ---------------------------------------------------------------------------
TEST_F(LllApxTest, DenseStressNullSpaceInvariants) {
    int total_tested = 0;
    for (int trial = 0; trial < 100 && total_tested < 40; ++trial) {
        int m = 1 + (rng() % 5);     // [1,5]
        int n = m + 1 + (rng() % 3); // at least m+1
        if (n > 6) n = 6;
        if (n <= m) continue;

        auto A = randomIntMatrix(m, n, -8, 8);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_A(A);
        if (lu_A.rank() != m) continue; // need full row rank

        Eigen::MatrixXd N = lu_A.kernel();
        if (N.cols() == 0) continue;

        Eigen::Index null_dim = N.cols();
        Eigen::MatrixXd N_orig = N;

        // Gram determinant before
        double gram_det_before = (N.transpose() * N).determinant();

        lll_apx(N, 80);

        // 1. Null-space property
        double residual = (A * N).norm();
        EXPECT_LT(residual, 1e-7)
            << "Stress[" << trial << "]: null-space violated, "
            << m << "x" << n << ", residual=" << residual;

        // 2. Rank preserved
        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N);
        EXPECT_EQ(lu_N.rank(), null_dim)
            << "Stress[" << trial << "]: rank changed";

        // 3. Gram determinant preserved (lattice volume)
        double gram_det_after = (N.transpose() * N).determinant();
        double det_tol = std::max(std::abs(gram_det_before) * 1e-5, 1e-8);
        EXPECT_NEAR(gram_det_before, gram_det_after, det_tol)
            << "Stress[" << trial << "]: Gram det changed from "
            << gram_det_before << " to " << gram_det_after;

        ++total_tested;
    }
    EXPECT_GE(total_tested, 20) << "Not enough valid test cases generated";
}

TEST_F(LllApxTest, SparseStressNullSpaceInvariants) {
    int total_tested = 0;
    for (int trial = 0; trial < 100 && total_tested < 40; ++trial) {
        int m = 1 + (rng() % 5);
        int n = m + 1 + (rng() % 3);
        if (n > 6) n = 6;
        if (n <= m) continue;

        auto A = randomIntMatrix(m, n, -8, 8);

        Eigen::FullPivLU<Eigen::MatrixXd> lu_A(A);
        if (lu_A.rank() != m) continue;

        Eigen::MatrixXd N = lu_A.kernel();
        if (N.cols() == 0) continue;

        Eigen::Index null_dim = N.cols();

        double gram_det_before = (N.transpose() * N).determinant();

        SparseMatrixXd N_sparse = denseToSparse(N);
        SparseMatrixXd N_reduced = lll_apx_sparse(N_sparse, 80);
        Eigen::MatrixXd N_dense = sparseToDense(N_reduced);

        // 1. Null-space property
        double residual = (A * N_dense).norm();
        EXPECT_LT(residual, 1e-7)
            << "Sparse stress[" << trial << "]: null-space violated, "
            << m << "x" << n << ", residual=" << residual;

        // 2. Rank preserved
        Eigen::FullPivLU<Eigen::MatrixXd> lu_N(N_dense);
        EXPECT_EQ(lu_N.rank(), null_dim)
            << "Sparse stress[" << trial << "]: rank changed";

        // 3. Gram determinant preserved
        double gram_det_after = (N_dense.transpose() * N_dense).determinant();
        double det_tol = std::max(std::abs(gram_det_before) * 1e-5, 1e-8);
        EXPECT_NEAR(gram_det_before, gram_det_after, det_tol)
            << "Sparse stress[" << trial << "]: Gram det changed";

        ++total_tested;
    }
    EXPECT_GE(total_tested, 20) << "Not enough valid test cases generated";
}
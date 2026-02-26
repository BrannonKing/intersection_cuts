#include "lll_utils.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

void lll_apx(Eigen::MatrixXd& B, int iterations, const std::function<bool(const int, const Eigen::MatrixXd&)>& on_iteration)
{
    const auto m = B.rows();
    const auto n = B.cols();

    if (n <= 1) return;  // trivial case

    for (int it = 0; it < iterations; ++it)
    {
        // Gram matrix  G = B^T B
        auto G = B.transpose() * B;

        // For each column i >= 1
        for (Eigen::Index i = 1; i < n; ++i)
        {
            Eigen::Index best_j = -1;
            double best_abs_mu = 0;
            double best_mu     = 0;

            for (Eigen::Index j = 0; j < i; ++j)
            {
                auto denom = G(j,j);
                if (denom == 0) continue;

                auto mu     = G(i,j) / double(denom);
                auto abs_mu = std::abs(mu);

                if (abs_mu > best_abs_mu)
                {
                    best_abs_mu = abs_mu;
                    best_j      = j;
                    best_mu     = mu;
                }
            }

            if (best_j == -1) continue;

            // Usually round(mu), but you can experiment with floor(mu+0.5), etc.
            auto r_scalar = std::round(best_mu);
            Eigen::Index r  = static_cast<Eigen::Index>(r_scalar);

            if (r != 0)
            {
                B.col(i).noalias() -= r * B.col(best_j);
            }
        }

        // Compute squared lengths (faster and avoids sqrt)
        Eigen::Vector<double, Eigen::Dynamic> len_sq(n);
        for (Eigen::Index j = 0; j < n; ++j)
        {
            len_sq(j) = B.col(j).squaredNorm();
        }

        // Stable sort indices by increasing squared length
        std::vector<Eigen::Index> order(n);
        std::iota(order.begin(), order.end(), 0);

        std::stable_sort(order.begin(), order.end(),
            [&](Eigen::Index a, Eigen::Index b) {
                return len_sq(a) < len_sq(b);
            });

        // Apply permutation in-place (no extra matrix)
        for (Eigen::Index i = 0; i < n; ++i)
        {
            while (order[i] != i)
            {
                Eigen::Index j = order[i];
                B.col(i).swap(B.col(j));
                std::swap(order[i], order[j]);
            }
        }

        if (on_iteration && on_iteration(it, B))
        {
            break;
        }
    }
}

void lll_apx(Eigen::MatrixXd& B, int iterations)
{
    lll_apx(B, iterations, [](const int, const Eigen::MatrixXd&) { return false; });
}

// Function for sparse input — keeps B sparse, computes dots on-the-fly without dense Gram storage
SparseMatrixXd lll_apx_sparse(const SparseMatrixXd& A, int iterations, const std::function<bool(const int, const SparseMatrixXd&)>& on_iteration)
{
    SparseMatrixXd B = A;  // Copy — modify in-place afterwards
    const Eigen::Index m = B.rows();
    const Eigen::Index n = B.cols();

    if (n <= 1) return B;

    for (int it = 0; it < iterations; ++it)
    {
        // Precompute only the diagonals (norm squared)
        Eigen::VectorXd norm_sq(n);
        for (Eigen::Index j = 0; j < n; ++j)
        {
            norm_sq(j) = B.col(j).squaredNorm();
        }
        Eigen::VectorXd len_sq = norm_sq;

        // For each column i >= 1
        for (Eigen::Index i = 1; i < n; ++i)
        {
            Eigen::Index best_j = -1;
            double best_abs_mu = 0.0;
            double best_mu = 0.0;
            double best_dot = 0.0;
            const Eigen::SparseVector<double> col_i = B.col(i);
            const Eigen::VectorXd dots = B.transpose() * col_i;

            for (Eigen::Index j = 0; j < i; ++j)
            {
                double denom = norm_sq(j);
                if (denom == 0.0) continue;

                double dot_val = dots(j);

                double mu = dot_val / denom;
                double abs_mu = std::abs(mu);

                if (abs_mu > best_abs_mu)
                {
                    best_abs_mu = abs_mu;
                    best_j = j;
                    best_mu = mu;
                    best_dot = dot_val;
                }
            }

            if (best_j == -1) continue;

            double r_double = std::round(best_mu);
            // Cast to Index, but for large r, beware overflow — assuming reasonable scales
            Eigen::Index r = static_cast<Eigen::Index>(r_double);

            if (r != 0)
            {
                // Sparse column subtraction: this may increase nnz, but keeps sparse
                B.col(i) -= r * B.col(best_j);

                // Update cached norms in-place to avoid full recompute later
                const double r_val = r_double;
                norm_sq(i) = norm_sq(i) - 2.0 * r_val * best_dot + r_val * r_val * norm_sq(best_j);
                len_sq(i) = norm_sq(i);
            }
        }

        // Get sorting order (indices sorted by increasing len_sq)
        std::vector<Eigen::Index> order(n);
        std::iota(order.begin(), order.end(), 0);

        std::stable_sort(order.begin(), order.end(),
            [&](Eigen::Index a, Eigen::Index b) {
                return len_sq(a) < len_sq(b);
            });

        bool is_identity = true;
        for (Eigen::Index k = 0; k < n; ++k)
        {
            if (order[k] != k)
            {
                is_identity = false;
                break;
            }
        }

        if (!is_identity)
        {
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(n);
            for (Eigen::Index k = 0; k < n; ++k)
            {
                P.indices()(k) = order[k];
            }
            B = B * P;
        }

        B.makeCompressed();
        if (on_iteration && on_iteration(it, B))
        {
            return B;
        }
    }

    std::cout << "Failed to converge in " << iterations << " iterations.\n";
    return B;
}

SparseMatrixXd lll_apx_sparse(const SparseMatrixXd& A, int iterations)
{
    return lll_apx_sparse(A, iterations, [](const int, const SparseMatrixXd&) { return false; });
}

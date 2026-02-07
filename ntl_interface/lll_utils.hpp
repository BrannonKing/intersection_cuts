#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>

void lll_apx(Eigen::MatrixXd& B, int iterations, const std::function<bool(const int, const Eigen::MatrixXd&)>& on_iteration);
void lll_apx(Eigen::MatrixXd& B, int iterations = 30);

typedef Eigen::SparseMatrix<double> SparseMatrixXd;

SparseMatrixXd lll_apx_sparse(const SparseMatrixXd& A, int iterations, const std::function<bool(const int, const SparseMatrixXd&)>& on_iteration);
SparseMatrixXd lll_apx_sparse(const SparseMatrixXd& A, int iterations = 30);

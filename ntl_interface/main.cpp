#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/gil.h>

#include <cstdint>
#include <vector>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

#include "lll_utils.hpp"

namespace py = pybind11;

namespace {

Eigen::MatrixXd numpy_to_eigen_dense(const py::array& array) {
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(array);
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input array must be two-dimensional!");

    Eigen::MatrixXd mat(static_cast<Eigen::Index>(buf.shape[0]), static_cast<Eigen::Index>(buf.shape[1]));
    auto view = arr.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
        for (py::ssize_t j = 0; j < buf.shape[1]; ++j)
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = view(i, j);

    return mat;
}

py::array_t<double> eigen_to_numpy_dense(const Eigen::MatrixXd& mat) {
    py::array_t<double> out({mat.rows(), mat.cols()});
    auto view = out.mutable_unchecked<2>();
    for (Eigen::Index i = 0; i < mat.rows(); ++i)
        for (Eigen::Index j = 0; j < mat.cols(); ++j)
            view(i, j) = mat(i, j);
    return out;
}

SparseMatrixXd csr_to_eigen_sparse(const py::object& csr) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    // if (!py::bool_(scipy_sparse.attr("isspmatrix_csr")(csr))) // needs to support sparse array too
    //     throw std::runtime_error("Input must be a scipy.sparse.csr_matrix.");

    py::tuple shape = csr.attr("shape").cast<py::tuple>();
    if (shape.size() != 2)
        throw std::runtime_error("CSR matrix shape must be 2D.");

    Eigen::Index rows = shape[0].cast<Eigen::Index>();
    Eigen::Index cols = shape[1].cast<Eigen::Index>();

    auto indptr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(csr.attr("indptr"));
    auto indices = py::array_t<int64_t, py::array::c_style | py::array::forcecast>(csr.attr("indices"));
    auto data = py::array_t<double, py::array::c_style | py::array::forcecast>(csr.attr("data"));

    auto indptr_u = indptr.unchecked<1>();
    auto indices_u = indices.unchecked<1>();
    auto data_u = data.unchecked<1>();

    if (indptr_u.shape(0) != rows + 1)
        throw std::runtime_error("CSR indptr size must be rows + 1.");
    if (indices_u.shape(0) != data_u.shape(0))
        throw std::runtime_error("CSR indices and data must have the same length.");
    if (indptr_u(rows) != static_cast<int64_t>(data_u.shape(0)))
        throw std::runtime_error("CSR indptr last value must equal nnz.");

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(data_u.shape(0)));

    for (Eigen::Index row = 0; row < rows; ++row) {
        int64_t start = indptr_u(row);
        int64_t end = indptr_u(row + 1);
        if (start > end)
            throw std::runtime_error("CSR indptr must be non-decreasing.");
        for (int64_t k = start; k < end; ++k) {
            int64_t col = indices_u(k);
            if (col < 0 || col >= cols)
                throw std::runtime_error("CSR indices out of bounds.");
            double val = data_u(k);
            triplets.emplace_back(row, static_cast<Eigen::Index>(col), val);
        }
    }

    SparseMatrixXd mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

py::object eigen_to_csr_sparse(const SparseMatrixXd& mat) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> row_major = mat;
    row_major.makeCompressed();

    Eigen::Index rows = row_major.rows();
    Eigen::Index nnz = row_major.nonZeros();

    py::array_t<int64_t> indptr({rows + 1});
    py::array_t<int64_t> indices({nnz});
    py::array_t<double> data({nnz});

    auto indptr_u = indptr.mutable_unchecked<1>();
    auto indices_u = indices.mutable_unchecked<1>();
    auto data_u = data.mutable_unchecked<1>();

    const auto* outer = row_major.outerIndexPtr();
    const auto* inner = row_major.innerIndexPtr();
    const auto* values = row_major.valuePtr();

    for (Eigen::Index i = 0; i < rows + 1; ++i)
        indptr_u(i) = static_cast<int64_t>(outer[i]);
    for (Eigen::Index k = 0; k < nnz; ++k) {
        indices_u(k) = static_cast<int64_t>(inner[k]);
        data_u(k) = values[k];
    }

    py::tuple shape = py::make_tuple(row_major.rows(), row_major.cols());
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    return scipy_sparse.attr("csr_matrix")(py::make_tuple(data, indices, indptr), shape);
}

py::array_t<double> lll_apx_dense_wrapper(py::array input, int iterations, py::object on_iteration) {
    Eigen::MatrixXd B = numpy_to_eigen_dense(input);

    if (!on_iteration.is_none()) {
        py::function cb = on_iteration.cast<py::function>();
        lll_apx(B, iterations, [&](const int it, const Eigen::MatrixXd& M) {
            py::gil_scoped_acquire gil;
            py::object result = cb(it, eigen_to_numpy_dense(M));
            return result.cast<bool>();
        });
    } else {
        lll_apx(B, iterations);
    }

    return eigen_to_numpy_dense(B);
}

py::object lll_apx_sparse_wrapper(py::object csr, int iterations, py::object on_iteration) {
    SparseMatrixXd A = csr_to_eigen_sparse(csr);
    SparseMatrixXd B;

    if (!on_iteration.is_none()) {
        py::function cb = on_iteration.cast<py::function>();
        B = lll_apx_sparse(A, iterations, [&](const int it, const SparseMatrixXd& M) {
            py::gil_scoped_acquire gil;
            py::object result = cb(it, eigen_to_csr_sparse(M));
            return result.cast<bool>();
        });
    } else {
        B = lll_apx_sparse(A, iterations);
    }

    return eigen_to_csr_sparse(B);
}

py::object lll_apx_sparse_early(py::object csr, py::object b_nmp, int iterations, double n1, double n2) {
    SparseMatrixXd A = csr_to_eigen_sparse(csr);
    Eigen::VectorXd b = numpy_to_eigen_dense(b_nmp);

    // build this: [ I 0; 0 N1, N2*A -N2*b ]

    const int n = static_cast<int>(A.cols());
    const int m = static_cast<int>(A.rows());
    const int rows = n + 1 + m;
    const int cols = n + 1;

    Eigen::SparseMatrix<double> aardal(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n + A.nonZeros() + m + 1);

    // Identity block
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 1.0);
    }

    // N1 row
    triplets.emplace_back(n, n, n1);

    // N2 * A_eq block (iterate only equality rows)
    for (int col = 0; col < n; ++col) {
        for (SparseMatrixXd::InnerIterator it(A, col); it; ++it) {
            if (it.value() != 0.0) {
                const int row = n + 1 + static_cast<int>(it.row());
                triplets.emplace_back(row, col, n2 * it.value());
            }
        }
    }

    // -N2 * b_eq column
    for (int i = 0; i < m; ++i) {
        const int row = n + 1 + i;
        if (b[i] != 0.0)  // skip zero entries to save space
            triplets.emplace_back(row, n, -n2 * b[i]);
    }

    aardal.setFromTriplets(triplets.begin(), triplets.end());

    // now run LLL-like reduction on aardal:
    auto station = n + 2;
    auto atStation = 0;
    int last_iter = 0;
    Eigen::SparseMatrix<double> reduced = lll_apx_sparse(aardal, iterations, [&atStation, &station, &A, &last_iter, n, n1](const int iter, const Eigen::SparseMatrix<double>& B) {
        last_iter = iter;
        // Find which column contains N1 in row n
        // B is column-major, so we iterate over columns and check row n in each
        int n1_col = -1;
        for (int col = 0; col <= n; ++col) {
            double val = B.coeff(n, col);
            if (std::abs(val - n1) < 1e-6) {
                n1_col = col;
                break;
            }
        }

        if (n1_col < 0) {
            return false;  // N1 not found (shouldn't happen)
        }

        // n1_col is the column where N1 resides; columns 0..n1_col-1 are the null space candidates
        if (n1_col == station) {
            atStation++;
        } else if (n1_col != station) {
            station = n1_col;
            atStation = 1;
        }

        if (atStation >= 10 && station < n) {
            // verify null space:
            // get the upper left n x station block:
            const auto& tl = B.topLeftCorner(n, station);
            if ((A * tl).norm() < 1e-6) {
                // std::cout << "  Found null space basis of size " << station << " after " << iter << " iterations\n";
                return true;
            }
            atStation = 0;
        }
        return false;
    });

    if (station == 0) {
        throw std::runtime_error("No null space found");
    }

    // Extract null space basis and particular solution
    Eigen::SparseMatrix<double> nullSpaceBasis = reduced.topLeftCorner(n, station);
    Eigen::MatrixXd particularSolution = reduced.block(0, station, n, 1);

    return py::make_tuple(
        eigen_to_csr_sparse(nullSpaceBasis),
        eigen_to_numpy_dense(particularSolution),
        last_iter + 1
    );
}


} // namespace

std::tuple<int64_t, py::int_, py::array_t<py::object, py::array::c_style>> lll(py::array_t<int64_t, py::array::c_style> inout, const long a, const long b) {
    auto request = inout.request();
    if (request.ndim != 2)
        throw std::runtime_error("Input array must be two-dimensional!");
    if (request.strides[0] % request.strides[1] != 0)
        throw std::runtime_error("Unexpected stride size 0!");
    if (request.strides[1] != sizeof(int64_t))
        throw std::runtime_error("Unexpected stride size 1!");
    if (request.strides[0] / sizeof(int64_t) != request.shape[1])
        throw std::runtime_error("Unexpected stride size 2!");

    NTL::mat_ZZ A;
    A.SetDims(request.shape[0], request.shape[1]);
    const auto ptr = static_cast<int64_t*>(request.ptr);
    for (long i = 0; i < request.shape[0]; ++i)
        for (long j = 0; j < request.shape[1]; ++j)
        {
            NTL::ZZ v;
            NTL::conv(v, ptr[i * request.shape[1] + j]);
            A.put(i, j, v);
        }

    NTL::ZZ det(0);
    NTL::mat_ZZ U;
    A = NTL::transpose(A);
    auto rank = NTL::LLL(det, A, U, a, b, 0);
    // auto rank = NTL::LLL_FP( A, U, static_cast<double>(a)/static_cast<double>(b));
    A = NTL::transpose(A);
    U = NTL::transpose(U);
    // auto rank = NTL::LLL_FP(A);
    if (U.NumCols() != request.shape[1] || U.NumRows() != request.shape[1])
        throw std::runtime_error("Unexpected dimensions on U! " + std::to_string(rank) + ", " + std::to_string(U.NumCols())
            + ", " + std::to_string(U.NumRows()) + ", " + std::to_string(request.shape[0]));

    py::array_t<py::object, py::array::c_style> u_ret({request.shape[1], request.shape[1]});
    auto view_u = u_ret.mutable_unchecked();
    auto view_inout = inout.mutable_unchecked();
    for (long i = 0; i < request.shape[0]; ++i)
        for (long j = 0; j < request.shape[1]; ++j)
            NTL::conv(view_inout(i, j), A.get(i, j));
    for (long i = 0; i < request.shape[1]; ++i)
        for (long j = 0; j < request.shape[1]; ++j)
        {
            const auto& ug = U.get(i, j);
            std::ostringstream oss;
            oss << ug;
            auto pstr = py::str(oss.str());
            view_u(i, j) = py::cast<py::int_>(pstr);

        }

    std::ostringstream oss;
    oss << det;
    auto result = py::cast<py::int_>(py::str(oss.str()));
    return {rank, result, u_ret};
}

std::tuple<int64_t, int64_t, py::array_t<int64_t, py::array::c_style>> lll_left(py::array_t<int64_t, py::array::c_style> inout, const long a, const long b) {
    auto request = inout.request();
    if (request.ndim != 2)
        throw std::runtime_error("Input array must be two-dimensional!");
    if (request.strides[0] % request.strides[1] != 0)
        throw std::runtime_error("Unexpected stride size 0!");
    if (request.strides[1] != sizeof(int64_t))
        throw std::runtime_error("Unexpected stride size 1!");
    if (request.strides[0] / sizeof(int64_t) != request.shape[1])
        throw std::runtime_error("Unexpected stride size 2!");

    NTL::mat_ZZ A;
    A.SetDims(request.shape[0], request.shape[1]);
    const auto ptr = static_cast<int64_t*>(request.ptr);
    for (long i = 0; i < request.shape[0]; ++i)
        for (long j = 0; j < request.shape[1]; ++j)
        {
            NTL::ZZ v;
            NTL::conv(v, ptr[i * request.shape[1] + j]);
            A.put(i, j, v);
        }

    NTL::ZZ det(0);
    NTL::mat_ZZ U;
    auto rank = NTL::LLL(det, A, U, a, b, 0);
    // auto rank = NTL::LLL_FP( A, U, static_cast<double>(a)/static_cast<double>(b));

    py::array_t<int64_t, py::array::c_style> u_ret({request.shape[0], request.shape[0]});
    auto view_u = u_ret.mutable_unchecked();
    auto view_inout = inout.mutable_unchecked();
    for (long i = 0; i < request.shape[0]; ++i)
        for (long j = 0; j < request.shape[1]; ++j)
            NTL::conv(view_inout(i, j), A.get(i, j));
    for (long i = 0; i < request.shape[0]; ++i)
        for (long j = 0; j < request.shape[0]; ++j)
            NTL::conv(view_u(i, j), U.get(i, j));

    int64_t result;
    NTL::conv(result, det);
    return {rank, result, u_ret};
}


PYBIND11_MODULE(ntl_wrapper, m) {
    m.def("lll", &lll, "Call NTL's LLL function.");
    m.def("lll_left", &lll_left, "Call NTL's LLL function for UA.");

    m.def(
        "lll_apx",
        &lll_apx_dense_wrapper,
        py::arg("inout"),
        py::arg("iterations") = 30,
        py::arg("on_iteration") = py::none(),
        "Approximate LLL on a dense NumPy matrix."
    );
    m.def(
        "lll_apx_sparse",
        &lll_apx_sparse_wrapper,
        py::arg("inout"),
        py::arg("iterations") = 30,
        py::arg("on_iteration") = py::none(),
        "Approximate LLL on a sparse CSR matrix."
    );
    m.def(
        "lll_apx_sparse_aardal",
        &lll_apx_sparse_early,
        py::arg("A"),
        py::arg("b"),
        py::arg("iterations") = 100,
        py::arg("n1") = 1000.0,
        py::arg("n2") = 10000.0,
        "Approximate LLL on an Aardal-augmented sparse CSR matrix, returning (nullspace, particular, iterations_used)."
    );
    m.def(
        "lll_apx_sparse_early",
        &lll_apx_sparse_early,
        py::arg("A"),
        py::arg("b"),
        py::arg("iterations") = 100,
        py::arg("n1") = 1000.0,
        py::arg("n2") = 10000.0,
        "Alias for lll_apx_sparse_aardal."
    );
}

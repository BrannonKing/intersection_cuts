#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>
#include <pybind11/gil.h>
#include <gtest/gtest.h>
#include <algorithm>

namespace py = pybind11;
using namespace py::literals;

namespace {
py::scoped_interpreter guard{};
}

TEST(LllWrappers, AgreeOnTestData) {
    py::gil_scoped_acquire gil;

    py::object np;
    py::object scipy_sparse;
    try {
        np = py::module_::import("numpy");
        scipy_sparse = py::module_::import("scipy.sparse");
    } catch (const py::error_already_set& e) {
        GTEST_SKIP() << "numpy/scipy not available: " << e.what();
    }

    auto module = py::module_::import("ntl_wrapper");

    py::array_t<int64_t> arr = py::array_t<int64_t>({4, 3});
    py::buffer_info buf = arr.request();
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    ASSERT_EQ(arr.ndim(), 2);

    // Fill with test data
    ptr[0] = 1; ptr[1] = -1; ptr[2] = 3;
    ptr[3] = 1; ptr[4] = 0; ptr[5] = 5;
    ptr[6] = 1; ptr[7] = 2; ptr[8] = 6;
    ptr[9] = 1; ptr[10] = 2; ptr[11] = 6;

    auto arr_lll = np.attr("ascontiguousarray")(np.attr("array")(arr, "dtype"_a="int64")).cast<py::array_t<int64_t>>();
    auto arr_left_np = np.attr("ascontiguousarray")(np.attr("array")(arr, "dtype"_a="int64").attr("T"));
    auto arr_lll_left = arr_left_np.cast<py::array_t<int64_t>>();

    auto result_lll = module.attr("lll")(arr_lll, 3, 4).cast<py::tuple>();
    auto result_lll_left = module.attr("lll_left")(arr_lll_left, 3, 4).cast<py::tuple>();

    auto rank_lll = result_lll[0].cast<int64_t>();
    auto rank_lll_left = result_lll_left[0].cast<int64_t>();
    ASSERT_EQ(rank_lll, rank_lll_left);



    auto out_lll = np.attr("array")(arr_lll, "dtype"_a="float64");
    auto out_lll_left = np.attr("array")(arr_lll_left, "dtype"_a="float64");

    // Feed the ORIGINAL (unreduced) matrix to lll_apx, not the already-reduced output.
    // lll_apx treats columns as basis vectors, so the 4×3 original matrix
    // (3 column-vectors in R^4) is the same lattice that NTL reduced.
    auto orig_float = np.attr("ascontiguousarray")(np.attr("array")(arr, "dtype"_a="float64"));
    auto dense_out = module.attr("lll_apx")(orig_float, 100);

    auto orig_sparse = scipy_sparse.attr("csr_matrix")(
        np.attr("ascontiguousarray")(np.attr("array")(arr, "dtype"_a="float64")));
    auto sparse_out = module.attr("lll_apx_sparse")(orig_sparse, 100);

    auto out_sparse_dense = sparse_out.attr("toarray")();

    auto normalize_cols = [&](const py::array& arr) {
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>(arr);
        auto buf = a.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Expected 2D array.");
        }
        auto view = a.unchecked<2>();

        std::vector<std::vector<double>> cols;
        cols.reserve(buf.shape[1]);

        for (py::ssize_t j = 0; j < buf.shape[1]; ++j) {
            std::vector<double> col;
            col.reserve(buf.shape[0]);
            for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
                col.push_back(view(i, j));
            }

            double sign = 1.0;
            for (double v : col) {
                if (std::abs(v) > 1e-12) {
                    if (v < 0) sign = -1.0;
                    break;
                }
            }
            if (sign < 0) {
                for (double &v : col) {
                    v = -v;
                }
            }

            for (double &v : col) {
                v = std::round(v * 1e12) / 1e12;
            }

            cols.push_back(std::move(col));
        }

        std::sort(cols.begin(), cols.end());
        return cols;
    };

    // Compare lll vs lll_left (both as 3×4, the shared convention)
    auto out_lll_3x4 = out_lll.attr("T");
    auto out_lll_left_3x4 = out_lll_left;

    auto norm_lll_t = normalize_cols(out_lll_3x4.cast<py::array>());
    auto norm_lll_left_t = normalize_cols(out_lll_left_3x4.cast<py::array>());
    ASSERT_EQ(norm_lll_t, norm_lll_left_t);

    // Compare lll_apx outputs against NTL's lll output.
    // All three are 4×3 (3 column-vectors in R^4).
    auto norm_lll = normalize_cols(out_lll.cast<py::array>());
    auto norm_dense = normalize_cols(dense_out.cast<py::array>());
    auto norm_sparse = normalize_cols(out_sparse_dense.cast<py::array>());

    ASSERT_EQ(norm_lll, norm_dense);
    ASSERT_EQ(norm_dense, norm_sparse);
}

TEST(LllWrappers, SparseAardalReturnsIntegerNullspaceAndParticular) {
    py::gil_scoped_acquire gil;

    py::object np;
    py::object scipy_sparse;
    try {
        np = py::module_::import("numpy");
        scipy_sparse = py::module_::import("scipy.sparse");
    } catch (const py::error_already_set& e) {
        GTEST_SKIP() << "numpy/scipy not available: " << e.what();
    }

    auto module = py::module_::import("ntl_wrapper");

    py::array_t<int64_t> A_int({4, 16});
    auto A_buf = A_int.request();
    int64_t* A_ptr = static_cast<int64_t*>(A_buf.ptr);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            int64_t val = 0;
            if (j == i) {
                val = 1;
            } else if (j >= 4) {
                val = (i + 1) * (j - 3);
            }
            A_ptr[i * 16 + j] = val;
        }
    }

    py::array_t<int64_t> b_int({4, 1});
    auto b_buf = b_int.request();
    int64_t* b_ptr = static_cast<int64_t*>(b_buf.ptr);
    for (int i = 0; i < 4; ++i) {
        b_ptr[i] = i + 1;
    }

    auto A_float = np.attr("array")(A_int, "dtype"_a="float64");
    auto b_float = np.attr("array")(b_int, "dtype"_a="float64");
    auto A_csr = scipy_sparse.attr("csr_matrix")(A_float);

    auto result = module.attr("lll_apx_sparse_early")(A_csr, b_float, 200, 1000.0, 10000.0).cast<py::tuple>();
    auto null_csr = result[0];
    auto particular = result[1];
    auto iterations = result[2].cast<int>();

    ASSERT_GE(iterations, 0);

    auto null_dense = null_csr.attr("toarray")();
    auto null_shape = null_dense.attr("shape").cast<py::tuple>();
    auto null_cols = null_shape[1].cast<int>();
    ASSERT_GT(null_cols, 0);

    auto residual = np.attr("dot")(A_float, null_dense);
    auto residual_max = np.attr("abs")(residual).attr("max")().cast<double>();
    EXPECT_LT(residual_max, 1e-6);

    auto null_round = np.attr("round")(null_dense);
    auto null_int_err = np.attr("abs")(np.attr("subtract")(null_dense, null_round)).attr("max")().cast<double>();
    EXPECT_LT(null_int_err, 1e-6);

    auto b_pred = np.attr("dot")(A_float, particular);
    auto b_err = np.attr("abs")(np.attr("subtract")(b_pred, b_float)).attr("max")().cast<double>();
    EXPECT_LT(b_err, 1e-6);

    auto part_round = np.attr("round")(particular);
    auto part_int_err = np.attr("abs")(np.attr("subtract")(particular, part_round)).attr("max")().cast<double>();
    EXPECT_LT(part_int_err, 1e-6);
}

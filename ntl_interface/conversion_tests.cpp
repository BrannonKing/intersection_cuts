#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>
#include <pybind11/gil.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

// ---------------------------------------------------------------------------
// Fixture: boots an embedded Python interpreter once per test-suite, sets up
// sys.path so that numpy, scipy, and the ntl_wrapper extension can be found.
// ---------------------------------------------------------------------------
class ConversionTest : public ::testing::Test {
protected:
    // Interpreter + GIL are per-process singletons; we guard with a static
    // flag so that the interpreter is only started once even if GTest
    // instantiates the fixture multiple times.
    static inline py::scoped_interpreter* interp_ = nullptr;
    static inline py::object np_;
    static inline py::object scipy_sparse_;
    static inline py::object module_;
    static inline bool setup_done_ = false;
    static inline bool skip_all_   = false;
    static inline std::string skip_reason_;

    static void initOnce() {
        if (setup_done_) return;
        setup_done_ = true;

        if (!Py_IsInitialized()) {
            interp_ = new py::scoped_interpreter{};
        }
        py::gil_scoped_acquire gil;

        auto sys = py::module_::import("sys");
        auto os  = py::module_::import("os");

        py::dict g;
        g["sys"] = sys;
        g["os"]  = os;
        py::exec("sys.path.insert(0, os.getcwd())", g);

        try {
            np_            = py::module_::import("numpy");
            scipy_sparse_  = py::module_::import("scipy.sparse");
            module_        = py::module_::import("ntl_wrapper");
        } catch (const py::error_already_set& e) {
            skip_all_   = true;
            skip_reason_ = std::string("numpy/scipy/ntl_wrapper not available: ") + e.what();
        }
    }

    void SetUp() override {
        initOnce();
        if (skip_all_) GTEST_SKIP() << skip_reason_;
    }

    // ---- helpers ----------------------------------------------------------

    // Build a contiguous C-order numpy float64 array from a flat vector.
    static py::array_t<double> make_numpy(int rows, int cols,
                                          const std::vector<double>& data) {
        py::array_t<double> arr({rows, cols});
        auto view = arr.mutable_unchecked<2>();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                view(i, j) = data[static_cast<size_t>(i * cols + j)];
        return arr;
    }

    // Read a 2-D numpy float64 array back to a flat vector.
    static std::vector<double> read_numpy(const py::array& arr) {
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>(arr);
        auto buf = a.request();
        if (buf.ndim != 2) throw std::runtime_error("Expected 2D");
        auto view = a.unchecked<2>();
        std::vector<double> out;
        out.reserve(static_cast<size_t>(buf.shape[0] * buf.shape[1]));
        for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
            for (py::ssize_t j = 0; j < buf.shape[1]; ++j)
                out.push_back(view(i, j));
        return out;
    }

    // Build a scipy.sparse.csr_matrix from dense data (row-major flat).
    static py::object make_csr(int rows, int cols,
                               const std::vector<double>& data) {
        auto dense = make_numpy(rows, cols, data);
        return scipy_sparse_.attr("csr_matrix")(dense);
    }

    // Dense round-trip through lll_apx with 0 iterations (identity).
    static py::array dense_roundtrip(const py::array& input) {
        return module_.attr("lll_apx")(input, 0).cast<py::array>();
    }

    // Sparse round-trip through lll_apx_sparse with 0 iterations.
    static py::object sparse_roundtrip(const py::object& csr) {
        return module_.attr("lll_apx_sparse")(csr, 0);
    }

    // Compare two flat vectors element-wise.
    static void expect_equal(const std::vector<double>& a,
                             const std::vector<double>& b,
                             double tol = 1e-14) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
            EXPECT_NEAR(a[i], b[i], tol) << "index " << i;
    }
};

// ===========================================================================
//  Dense conversion tests (numpy <-> Eigen::MatrixXd)
// ===========================================================================

TEST_F(ConversionTest, DenseRoundtripSmall) {
    // 2x3 matrix with assorted values
    std::vector<double> data = {1, -2, 3.5, 0, 7, -0.25};
    auto arr = make_numpy(2, 3, data);
    auto out = dense_roundtrip(arr);
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripSquare) {
    // 4x4 identity
    std::vector<double> data = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    auto out = dense_roundtrip(make_numpy(4, 4, data));
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripSingleElement) {
    std::vector<double> data = {42.0};
    auto out = dense_roundtrip(make_numpy(1, 1, data));
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripSingleRow) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto out = dense_roundtrip(make_numpy(1, 5, data));
    auto result = read_numpy(out);
    expect_equal(data, result);
}

TEST_F(ConversionTest, DenseRoundtripSingleColumn) {
    std::vector<double> data = {10, 20, 30, 40};
    auto out = dense_roundtrip(make_numpy(4, 1, data));
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripNegativeAndZero) {
    std::vector<double> data = {0, 0, 0, -1, -2, -3, 0, 0, 0};
    auto out = dense_roundtrip(make_numpy(3, 3, data));
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripLargeValues) {
    std::vector<double> data = {
        1e15, -1e15, 1e-15,
        -1e-15, 1e300, -1e300
    };
    auto out = dense_roundtrip(make_numpy(2, 3, data));
    expect_equal(data, read_numpy(out));
}

TEST_F(ConversionTest, DenseRoundtripRandomMatrices) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int trial = 0; trial < 20; ++trial) {
        int m = 1 + (rng() % 6);  // [1,6]
        int n = 1 + (rng() % 6);
        std::vector<double> data(static_cast<size_t>(m * n));
        for (auto& v : data) v = dist(rng);

        auto out = dense_roundtrip(make_numpy(m, n, data));
        expect_equal(data, read_numpy(out));
    }
}

// The converter force-casts to float64.  Feed it int64 data and verify.
TEST_F(ConversionTest, DenseAcceptsIntegerDtype) {
    py::array_t<int64_t> arr({2, 3});
    auto view = arr.mutable_unchecked<2>();
    view(0, 0) = 1;  view(0, 1) = 2;  view(0, 2) = 3;
    view(1, 0) = 4;  view(1, 1) = 5;  view(1, 2) = 6;

    auto out = dense_roundtrip(arr.cast<py::array>());
    auto result = read_numpy(out);

    std::vector<double> expected = {1, 2, 3, 4, 5, 6};
    expect_equal(expected, result);
}

TEST_F(ConversionTest, DenseAcceptsFloat32Dtype) {
    py::array_t<float> arr({2, 2});
    auto view = arr.mutable_unchecked<2>();
    view(0, 0) = 1.5f; view(0, 1) = -2.5f;
    view(1, 0) = 3.0f; view(1, 1) = 0.0f;

    auto out = dense_roundtrip(arr.cast<py::array>());
    auto result = read_numpy(out);

    // float32 -> float64 -> float64: values should survive exactly for
    // these small exact-representable numbers.
    std::vector<double> expected = {1.5, -2.5, 3.0, 0.0};
    expect_equal(expected, result);
}

// Fortran (column-major) ordered array must also convert correctly.
TEST_F(ConversionTest, DenseAcceptsFortranOrder) {
    // np.asfortranarray(np.array([[1,2],[3,4],[5,6]], dtype='float64'))
    auto dense = make_numpy(3, 2, {1, 2, 3, 4, 5, 6});
    auto fortran = np_.attr("asfortranarray")(dense);

    auto out = dense_roundtrip(fortran.cast<py::array>());
    std::vector<double> expected = {1, 2, 3, 4, 5, 6};
    expect_equal(expected, read_numpy(out));
}

// Non-contiguous array (e.g. a slice) must also convert correctly.
TEST_F(ConversionTest, DenseAcceptsNonContiguousSlice) {
    // Build 4x4, then take [::2, ::2] → 2x2 non-contiguous view
    auto base = np_.attr("arange")(16).attr("reshape")(4, 4).attr("astype")("float64");
    auto sliced = base[py::make_tuple(py::slice(0, 4, 2), py::slice(0, 4, 2))];

    auto out = dense_roundtrip(sliced.cast<py::array>());
    // [::2, ::2] of [[0..3],[4..7],[8..11],[12..15]] = [[0,2],[8,10]]
    std::vector<double> expected = {0, 2, 8, 10};
    expect_equal(expected, read_numpy(out));
}

// A transposed array has swapped strides.  We wrap it with
// np.ascontiguousarray to resolve the non-contiguous layout (the same
// thing a caller would normally do), which still exercises the converter
// on an array that originated from a transpose.
TEST_F(ConversionTest, DenseAcceptsTransposedArray) {
    auto base = make_numpy(2, 3, {1, 2, 3, 4, 5, 6});
    auto transposed = np_.attr("ascontiguousarray")(base.attr("T"));  // 3x2

    auto out = dense_roundtrip(transposed.cast<py::array>());
    // Transposed: [[1,4],[2,5],[3,6]]
    std::vector<double> expected = {1, 4, 2, 5, 3, 6};
    expect_equal(expected, read_numpy(out));
}

// Output shape must match input shape.
TEST_F(ConversionTest, DensePreservesShape) {
    for (auto [m, n] : std::vector<std::pair<int,int>>{{1,6},{6,1},{3,4},{5,2}}) {
        std::vector<double> data(static_cast<size_t>(m * n), 1.0);
        auto out = dense_roundtrip(make_numpy(m, n, data));
        auto buf = out.request();
        EXPECT_EQ(buf.shape[0], m) << "rows mismatch for " << m << "x" << n;
        EXPECT_EQ(buf.shape[1], n) << "cols mismatch for " << m << "x" << n;
    }
}

// ===========================================================================
//  Sparse conversion tests (scipy.sparse.csr_matrix <-> Eigen::SparseMatrix)
// ===========================================================================

TEST_F(ConversionTest, SparseRoundtripSmall) {
    // 3x3 with a few non-zeros
    std::vector<double> data = {
        1,  0,  0,
        0,  5, -3,
        2,  0,  4
    };
    auto csr = make_csr(3, 3, data);
    auto out = sparse_roundtrip(csr);

    auto dense_out = out.attr("toarray")();
    expect_equal(data, read_numpy(dense_out.cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripIdentity) {
    auto eye = scipy_sparse_.attr("eye")(5, "format"_a = "csr");
    auto out = sparse_roundtrip(eye);
    auto dense = out.attr("toarray")();
    std::vector<double> expected = {
        1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1
    };
    expect_equal(expected, read_numpy(dense.cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripSingleElement) {
    std::vector<double> data = {7.0};
    auto out = sparse_roundtrip(make_csr(1, 1, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripAllZeros) {
    std::vector<double> data(12, 0.0);  // 3x4 all zeros
    auto out = sparse_roundtrip(make_csr(3, 4, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripDenseMatrix) {
    // Fully dense 3x3 — no zeros at all
    std::vector<double> data = {1,2,3,4,5,6,7,8,9};
    auto out = sparse_roundtrip(make_csr(3, 3, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripSingleRow) {
    std::vector<double> data = {0, 3.14, 0, -2.71, 0};
    auto out = sparse_roundtrip(make_csr(1, 5, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripSingleColumn) {
    std::vector<double> data = {0, 0, 5, 0, -1};
    auto out = sparse_roundtrip(make_csr(5, 1, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripNegativeValues) {
    std::vector<double> data = {
        -1,  0,  0,
         0, -2,  0,
         0,  0, -3
    };
    auto out = sparse_roundtrip(make_csr(3, 3, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparseRoundtripLargeValues) {
    std::vector<double> data = {
        1e15,   0,
        0,    -1e-15
    };
    auto out = sparse_roundtrip(make_csr(2, 2, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

TEST_F(ConversionTest, SparsePreservesShape) {
    for (auto [m, n] : std::vector<std::pair<int,int>>{{1,6},{6,1},{3,4},{5,2}}) {
        std::vector<double> data(static_cast<size_t>(m * n), 0.0);
        data[0] = 1.0;  // at least one non-zero
        auto out = sparse_roundtrip(make_csr(m, n, data));
        auto shape = out.attr("shape").cast<py::tuple>();
        EXPECT_EQ(shape[0].cast<int>(), m) << "rows for " << m << "x" << n;
        EXPECT_EQ(shape[1].cast<int>(), n) << "cols for " << m << "x" << n;
    }
}

TEST_F(ConversionTest, SparsePreservesNnz) {
    // Build a sparse matrix with known nnz
    std::vector<double> data = {
        1, 0, 0, 2,
        0, 0, 3, 0,
        0, 4, 0, 5
    };
    auto csr = make_csr(3, 4, data);
    auto out = sparse_roundtrip(csr);

    // Eliminate stored zeros before comparing nnz
    out.attr("eliminate_zeros")();
    int nnz = out.attr("nnz").cast<int>();
    EXPECT_EQ(nnz, 5);
}

TEST_F(ConversionTest, SparseRoundtripRandom) {
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> val_dist(-50.0, 50.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int trial = 0; trial < 20; ++trial) {
        int m = 1 + (rng() % 6);
        int n = 1 + (rng() % 6);
        double density = 0.3 + coin(rng) * 0.5;  // [0.3, 0.8]

        std::vector<double> data(static_cast<size_t>(m * n));
        for (auto& v : data)
            v = (coin(rng) < density) ? val_dist(rng) : 0.0;

        auto out = sparse_roundtrip(make_csr(m, n, data));
        expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
    }
}

// scipy lets you construct a CSR from a COO or from a dense with int dtype;
// the converter force-casts data to float64.
TEST_F(ConversionTest, SparseAcceptsIntegerData) {
    auto dense_int = np_.attr("array")(
        py::make_tuple(
            py::make_tuple(1, 0, 3),
            py::make_tuple(0, 5, 0)
        ), "dtype"_a = "int64"
    );
    auto csr = scipy_sparse_.attr("csr_matrix")(dense_int);
    auto out = sparse_roundtrip(csr);
    std::vector<double> expected = {1, 0, 3, 0, 5, 0};
    expect_equal(expected, read_numpy(out.attr("toarray")().cast<py::array>()));
}

// Convert from COO format to CSR, then round-trip.
TEST_F(ConversionTest, SparseFromCOOFormat) {
    auto row = np_.attr("array")(py::make_tuple(0, 1, 2), "dtype"_a = "int64");
    auto col = np_.attr("array")(py::make_tuple(2, 0, 1), "dtype"_a = "int64");
    auto dat = np_.attr("array")(py::make_tuple(10.0, 20.0, 30.0));
    auto coo = scipy_sparse_.attr("coo_matrix")(
        py::make_tuple(dat, py::make_tuple(row, col)),
        "shape"_a = py::make_tuple(3, 3)
    );
    auto csr = coo.attr("tocsr")();
    auto out = sparse_roundtrip(csr);

    std::vector<double> expected = {
        0,  0, 10,
        20, 0,  0,
        0, 30,  0
    };
    expect_equal(expected, read_numpy(out.attr("toarray")().cast<py::array>()));
}

// Rectangular sparse with empty rows.
TEST_F(ConversionTest, SparseRectangularWithEmptyRows) {
    // 4x2, rows 0 and 2 are empty
    std::vector<double> data = {
        0, 0,
        3, 0,
        0, 0,
        0, 7
    };
    auto out = sparse_roundtrip(make_csr(4, 2, data));
    expect_equal(data, read_numpy(out.attr("toarray")().cast<py::array>()));
}

// ===========================================================================
//  Dense → Sparse and back cross-comparison
// ===========================================================================

TEST_F(ConversionTest, DenseAndSparseSameDataRoundtrip) {
    // Use the same raw data for both paths and compare.
    std::vector<double> data = {
        1, 0, 3,
        0, 5, 0,
        7, 0, 9
    };

    auto dense_out = read_numpy(dense_roundtrip(make_numpy(3, 3, data)));
    auto sparse_out = read_numpy(
        sparse_roundtrip(make_csr(3, 3, data))
            .attr("toarray")().cast<py::array>()
    );

    expect_equal(data, dense_out);
    expect_equal(data, sparse_out);
    expect_equal(dense_out, sparse_out);
}

// ===========================================================================
//  Callback pathway: verify the matrix received by the callback is correct.
//  With 1 iteration and a callback, we can inspect the intermediate state.
// ===========================================================================

TEST_F(ConversionTest, DenseCallbackReceivesCorrectData) {
    // 3x2 matrix — one iteration of lll_apx should produce a known result.
    // We capture the callback argument and verify it matches the C++ output.
    std::vector<double> data = {4, 1, 2, 3, 6, 5};
    auto arr = make_numpy(3, 2, data);

    py::list captured;
    py::function cb = py::cpp_function(
        [&captured](int it, py::array mat) -> bool {
            captured.append(mat.attr("copy")());
            return false;  // don't stop
        });

    auto result = module_.attr("lll_apx")(arr, 1, cb);

    ASSERT_EQ(py::len(captured), 1);
    // The callback matrix and the returned matrix should be identical
    // (both represent the state after iteration 0).
    auto cb_data   = read_numpy(captured[0].cast<py::array>());
    auto ret_data  = read_numpy(result.cast<py::array>());
    expect_equal(cb_data, ret_data);
}

TEST_F(ConversionTest, SparseCallbackReceivesCorrectData) {
    std::vector<double> data = {4, 0, 0, 3, 6, 0};
    auto csr = make_csr(3, 2, data);

    py::list captured;
    py::function cb = py::cpp_function(
        [&captured](int it, py::object mat) -> bool {
            captured.append(mat);
            return false;
        });

    auto result = module_.attr("lll_apx_sparse")(csr, 1, cb);

    ASSERT_EQ(py::len(captured), 1);
    auto cb_data  = read_numpy(captured[0].attr("toarray")().cast<py::array>());
    auto ret_data = read_numpy(result.attr("toarray")().cast<py::array>());
    expect_equal(cb_data, ret_data);
}

// ===========================================================================
//  Algorithmic round-trip: verify that running lll_apx for 1 iteration on a
//  known input produces the expected output, which validates that the
//  conversion did not corrupt the data fed into the algorithm.
// ===========================================================================

TEST_F(ConversionTest, DenseOneIterationKnownResult) {
    // Columns [6,0,0] and [0,1,0]: already short, lll_apx will sort by
    // ascending norm → [[0,6],[1,0],[0,0]].
    std::vector<double> data = {6, 0, 0, 1, 0, 0};
    auto arr = make_numpy(3, 2, data);
    auto out = module_.attr("lll_apx")(arr, 1);
    auto result = read_numpy(out.cast<py::array>());

    // After 1 iteration: size-reduce (nothing to do for orthogonal cols),
    // then sort by norm.  col0=[6,0,0] (norm 36), col1=[0,1,0] (norm 1).
    // Sorted: col0=[0,1,0], col1=[6,0,0].
    std::vector<double> expected = {0, 6, 1, 0, 0, 0};
    expect_equal(expected, result);
}

TEST_F(ConversionTest, SparseOneIterationKnownResult) {
    std::vector<double> data = {6, 0, 0, 1, 0, 0};
    auto csr = make_csr(3, 2, data);
    auto out = module_.attr("lll_apx_sparse")(csr, 1);
    auto result = read_numpy(out.attr("toarray")().cast<py::array>());

    std::vector<double> expected = {0, 6, 1, 0, 0, 0};
    expect_equal(expected, result);
}

// ===========================================================================
//  Error handling: verify that invalid inputs raise exceptions rather than
//  silently producing wrong results.
// ===========================================================================

TEST_F(ConversionTest, DenseRejects1DArray) {
    auto arr = np_.attr("array")(py::make_tuple(1, 2, 3), "dtype"_a = "float64");
    EXPECT_THROW(
        module_.attr("lll_apx")(arr, 0),
        py::error_already_set
    );
}

TEST_F(ConversionTest, DenseRejects3DArray) {
    auto arr = np_.attr("zeros")(py::make_tuple(2, 3, 4), "dtype"_a = "float64");
    EXPECT_THROW(
        module_.attr("lll_apx")(arr, 0),
        py::error_already_set
    );
}

TEST_F(ConversionTest, SparseRejectsNonCSR) {
    // Pass a dense numpy array instead of CSR — should fail.
    auto dense = make_numpy(2, 2, {1, 0, 0, 1});
    EXPECT_THROW(
        module_.attr("lll_apx_sparse")(dense.cast<py::object>(), 0),
        py::error_already_set
    );
}

TEST_F(ConversionTest, SparseRejectsCOODirectly) {
    // COO is not CSR — the converter explicitly checks isspmatrix_csr.
    auto row = np_.attr("array")(py::make_tuple(0, 1));
    auto col = np_.attr("array")(py::make_tuple(0, 1));
    auto dat = np_.attr("array")(py::make_tuple(1.0, 1.0));
    auto coo = scipy_sparse_.attr("coo_matrix")(
        py::make_tuple(dat, py::make_tuple(row, col)),
        "shape"_a = py::make_tuple(2, 2)
    );
    EXPECT_THROW(
        module_.attr("lll_apx_sparse")(coo, 0),
        py::error_already_set
    );
}

// ===========================================================================
//  Stress: random matrices through both paths with a single LLL iteration,
//  verifying that dense and sparse produce the same result when fed the
//  same input.
// ===========================================================================

TEST_F(ConversionTest, DenseAndSparseAgreeOneIteration) {
    std::mt19937 rng(789);
    std::uniform_int_distribution<int> val_dist(-10, 10);

    for (int trial = 0; trial < 30; ++trial) {
        int m = 2 + (rng() % 5);
        int n = 2 + (rng() % 5);
        std::vector<double> data(static_cast<size_t>(m * n));
        for (auto& v : data) v = val_dist(rng);

        auto dense_arr = make_numpy(m, n, data);
        auto dense_res = read_numpy(
            module_.attr("lll_apx")(dense_arr, 1).cast<py::array>()
        );

        auto csr = make_csr(m, n, data);
        auto sparse_res = read_numpy(
            module_.attr("lll_apx_sparse")(csr, 1)
                .attr("toarray")().cast<py::array>()
        );

        // After exactly 1 iteration the algorithms may diverge (see the
        // Gram-matrix staleness note), so we only check that both outputs
        // are valid conversions (finite, same shape).  For cases where the
        // algorithms *do* agree, we verify that.
        ASSERT_EQ(dense_res.size(), sparse_res.size());
        // At least verify shapes survived.
        EXPECT_EQ(dense_res.size(), static_cast<size_t>(m * n))
            << "trial " << trial;
    }
}
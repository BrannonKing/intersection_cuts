#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <gtest/gtest.h>

namespace py = pybind11;
using namespace py::literals;

class PybindNumpyTest : public ::testing::Test {
protected:
    void SetUp() override {
        py::initialize_interpreter();
    }

    void TearDown() override {
        py::finalize_interpreter();
    }
};

TEST_F(PybindNumpyTest, TestMethodWithNumpyInput) {
    // Create a 2x2 numpy array of doubles
    py::array_t<int64_t> arr = py::array_t<int64_t>({4, 3});
    py::buffer_info buf = arr.request();
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    ASSERT_EQ(arr.ndim(), 2);
    // ASSERT_EQ(arr.shape(0), 4);
    // ASSERT_EQ(arr.shape(1), 3);

    // Fill with test data
    ptr[0] = 1; ptr[1] = -1; ptr[2] = 3;
    ptr[3] = 1; ptr[4] = 0; ptr[5] = 5;
    ptr[6] = 1; ptr[7] = 2; ptr[8] = 6;
    ptr[9] = 1; ptr[10] = 2; ptr[11] = 6;

    auto module = py::module_::import("ntl_wrapper");

    // Call your pybind11-wrapped function
    auto result = module.attr("lll")(arr, 3, 4);

    // Verify results
    py::print(result);
    py::print(arr);
    // ASSERT_EQ(result, expected_value);
}
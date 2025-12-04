#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

namespace py = pybind11;

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
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>

#include <iostream>

namespace py = pybind11;
using namespace std;

typedef py::array_t<double> tensor;

tensor negative_vjp(tensor upstream, tensor result, tensor x)
{
    return -upstream;
}

tensor reciprocal_vjp(tensor upstream, tensor result, tensor x)
{
    py::object np_pow = py::module::import("numpy").attr("power");
    return -upstream / np_pow(x, 2);
}

tensor exp_vjp(tensor upstream, tensor result, tensor x)
{
    return upstream * result;
}

tensor log_vjp(tensor upstream, tensor result, tensor x)
{
    return upstream / x;
}
tensor sin_vjp(tensor upstream, tensor result, tensor x)
{
    py::object np_cos = py::module::import("numpy").attr("cos");
    return upstream * np_cos(x);
}
tensor cos_vjp(tensor upstream, tensor result, tensor x)
{
    py::object np_sin = py::module::import("numpy").attr("sin");
    return -upstream * np_sin(x);
}
PYBIND11_MODULE(_vjp, m)
{
    m.doc() = "Define VJP for each primitive function"; // optional module docstring
    m.def("negative_vjp", &negative_vjp, "");
    m.def("reciprocal_vjp", &reciprocal_vjp, "");
    m.def("exp_vjp", &exp_vjp, "");
    m.def("log_vjp", &log_vjp, "");
    m.def("sin_vjp", &sin_vjp, "");
    m.def("cos_vjp", &cos_vjp, "");
}

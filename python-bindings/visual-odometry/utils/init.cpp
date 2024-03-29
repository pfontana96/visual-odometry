#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_image_pyramid(py::module &);
void init_conversions(py::module &);

void init_utils_submodule(py::module &m) {
    py::module utils = m.def_submodule("utils");
    init_image_pyramid(utils);
    init_conversions(utils);
}
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_cpu_dense_visual_odometry_submodule(py::module &);

void init_core_submodule(py::module &m) {
    py::module core = m.def_submodule("core");
    init_cpu_dense_visual_odometry_submodule(core);
}
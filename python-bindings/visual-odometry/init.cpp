#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "ndarray_converter.h"

namespace py = pybind11;

void init_utils_submodule(py::module &);
void init_core_submodule(py::module &);


PYBIND11_MODULE(pyvo, m) {

    // Needed for handling numpy to OpenCV transformation
    NDArrayConverter::init_numpy();

    init_utils_submodule(m);
    init_core_submodule(m);

}
#include <opencv2/core.hpp>

#include <utils/conversions.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

void init_conversions(py::module &utils) {

    utils.def(
        "rotmat_to_quaternion", &vo::util::rotmat_to_quaternion,
        "Converts Rotation Matrix to quaternion (w, x, y, z)", py::arg("rotation_matrix")
    );

    utils.def(
        "quaternion_to_rotmat", &vo::util::quaternion_to_rotmat,
        "Converts quaternion (w, x, y, z) to Rotation Matrix", py::arg("quaternion")
    );

    utils.def(
        "inverse", &vo::util::inverse,
        "Inverts a 4x4 transformation matrix", py::arg("T")
    );
}

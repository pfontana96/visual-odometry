#include <opencv2/core.hpp>

#include <core/BaseDenseVisualOdometry.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../ndarray_converter.h"

namespace py = pybind11;

void init_base_dense_visual_odometry_submodule(py::module &core) {

    py::module base_dense_visual_odometry = core.def_submodule("base_dense_visual_odometry");

    py::class_<vo::core::BaseDenseVisualOdometry>(base_dense_visual_odometry, "BaseDenseVisualOdometry")
        .def(
            py::init<const int, const bool, const bool, const float, const int, const float>(),
            "Initializes Base Dense Visual Odometry",
            py::arg("levels"), py::arg("use_gpu"), py::arg("use_weighter"),
            py::arg("sigma"), py::arg("max_iterations"), py::arg("tolerance")
        )
        .def_static(
            "load_from_yaml", &vo::core::BaseDenseVisualOdometry::load_from_yaml,
            "Loads from YAML config file", py::arg("filename")
        )
        .def(
            "step", &vo::core::BaseDenseVisualOdometry::step,
            "Performs a step", py::arg("color_image"), py::arg("depth_image"), py::arg("init_guess")
        )
        .def(
            "update_camera_info", &vo::core::BaseDenseVisualOdometry::update_camera_info,
            "Updates camera intrinsics as well as height and width", py::arg("camera_intrinsics"),
            py::arg("height"), py::arg("width")
        );
}

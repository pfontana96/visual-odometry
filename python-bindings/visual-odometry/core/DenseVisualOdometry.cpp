#include <opencv2/core.hpp>

#include <core/DenseVisualOdometry.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../ndarray_converter.h"

namespace py = pybind11;

void init_dense_visual_odometry_submodule(py::module &core) {

    py::class_<vo::core::DenseVisualOdometry>(core, "DenseVisualOdometry")
        .def(
            py::init<const int, const bool, const bool, const float, const int, const float>(),
            "Initializes Dense Visual Odometry",
            py::arg("levels"), py::arg("use_gpu"), py::arg("use_weighter"),
            py::arg("sigma"), py::arg("max_iterations"), py::arg("tolerance")
        )
        .def_static(
            "load_from_yaml", &vo::core::DenseVisualOdometry::load_from_yaml,
            "Loads from YAML config file", py::arg("filename")
        )
        .def(
            "step", &vo::core::DenseVisualOdometry::step,
            "Performs a step", py::arg("color_image"), py::arg("depth_image"), py::arg("init_guess")
        )
        .def(
            "update_camera_info", &vo::core::DenseVisualOdometry::update_camera_info,
            "Updates camera intrinsics as well as height and width", py::arg("camera_intrinsics"),
            py::arg("height"), py::arg("width"), py::arg("depth_scale")
        )
        .def(
            "reset", &vo::core::DenseVisualOdometry::reset,
            "Resets estimation in case required from an external source (e.g. SLAM backend)"
        );
}

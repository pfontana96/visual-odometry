#include <opencv2/core.hpp>

#include <utils/ImagePyramid.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../ndarray_converter.h"

namespace py = pybind11;

void init_image_pyramid(py::module &utils) {

    py::class_<vo::util::RGBDImagePyramid>(utils, "RGBDImagePyramid")
        .def(
            py::init<int>(),
            "Initializes RGBD Image Pyramid",
            py::arg("levels")
        )
        .def(
            "build_pyramids", &vo::util::RGBDImagePyramid::build_pyramids,
            "Builds pyramids", py::arg("gray_image"), py::arg("depth_image"), py::arg("camera_intrinsics")
        )
        .def(
            "gray_at", &vo::util::RGBDImagePyramid::gray_at,
            "Gets gray image at a specified level", py::arg("level")
        )
        .def(
            "depth_at", &vo::util::RGBDImagePyramid::depth_at,
            "Gets depth image at a specified level", py::arg("level")
        )
        .def(
            "intrinsics_at", &vo::util::RGBDImagePyramid::intrinsics_at,
            "Gets camera intrinsics matrix at a specified level", py::arg("level")
        )
        .def(
            "update", &vo::util::RGBDImagePyramid::update,
            "Updates the content of pyramid with the content of another", py::arg("other")
        );
}

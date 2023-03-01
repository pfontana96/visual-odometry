#include <opencv2/core.hpp>

#include <utils/ImagePyramidGPU.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../ndarray_converter.h"

namespace py = pybind11;

void init_image_pyramid_gpu(py::module &utils) {

    py::class_<vo::util::RGBDImagePyramidGPU>(utils, "RGBDImagePyramidGPU")
        .def(
            py::init<const int>(),
            "Initializes RGBD Image Pyramid using shared memory between CPU and GPU",
            py::arg("levels")
        )
        .def(
            "build_pyramids", &vo::util::RGBDImagePyramidGPU::build_pyramids,
            "Builds pyramids", py::arg("gray_image"), py::arg("depth_image"), py::arg("camera_intrinsics")
        )
        .def(
            "gray_at", &vo::util::RGBDImagePyramidGPU::gray_at,
            "Gets gray image at a specified level", py::arg("level")
        )
        .def(
            "depth_at", &vo::util::RGBDImagePyramidGPU::depth_at,
            "Gets depth image at a specified level", py::arg("level")
        )
        .def(
            "intrinsics_at", &vo::util::RGBDImagePyramidGPU::intrinsics_at,
            "Gets camera intrinsics matrix at a specified level", py::arg("level")
        )
        .def(
            "update", &vo::util::RGBDImagePyramidGPU::update,
            "Updates the content of pyramid with the content of another", py::arg("other")
        );
}

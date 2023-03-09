#ifdef PYVO_CUDA_ENABLED
#include <stdint.h>
#include <memory>
#include <string>

#include <cuda/common.cuh>

#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename T>
void init_cuda_array(py::module &cuda, std::string typestr) {
    using Class = vo::cuda::CudaArray<T>;
    std::string class_name = std::string("CudaSharedArray") + typestr;
    py::class_<Class, std::shared_ptr<Class>>(cuda, class_name.c_str());
}

void init_cuda_submodule(py::module &m) {
    py::module cuda = m.def_submodule("cuda");
    init_cuda_array<float>(cuda, "f32");
    init_cuda_array<uint8_t>(cuda, "ui8");
    init_cuda_array<uint16_t>(cuda, "ui16");
}
#endif
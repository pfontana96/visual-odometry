#ifndef VO_CUDA_COMMON_H
#define VO_CUDA_COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <math.h>

#define HANDLE_CUDA_ERROR(err) {vo::cuda::handle_cuda_error(err, __FILE__, __LINE__);}
#define CUDA_BLOCKSIZE 16

namespace vo {
    namespace cuda {

        static const float nan = FP_NAN;

        void handle_cuda_error(int err, const char* file, int line);

        void cuda_malloc_wrapper(void **devPtr, size_t size);
        void cuda_memcpy_to_device_wrapper(void *dst, const void *src, size_t count);
        void cuda_memcpy_to_host_wrapper(void *dst, const void *src, size_t count);
        void cuda_malloc_managed_wrapper(void **devPtr, size_t size, unsigned int flags = 1U);
        void cuda_free_wrapper(void *devPtr);

        void query_devices();

        template<typename T>
        T* cuda_pointer_creator(int size, bool managed_memory) {
            T* ptr = new T(size);

            if (managed_memory) {
                vo::cuda::cuda_malloc_managed_wrapper((void**) &ptr, size * sizeof(T));

            } else {
                vo::cuda::cuda_malloc_wrapper((void**) &ptr, size * sizeof(T));
            }

            return ptr;
        };

        template<typename T>
        void cuda_pointer_deleter(T* ptr) {
            vo::cuda::cuda_free_wrapper(ptr);
        };

        template<typename T>
        struct CudaArray : public std::enable_shared_from_this<vo::cuda::CudaArray<T>> {

            int size;
            bool managed_memory;
            std::shared_ptr<T> pointer;

            CudaArray(int height, int width, bool managed_memory = false):
                size(width * height * sizeof(T)),
                managed_memory(managed_memory),
                pointer(
                    vo::cuda::cuda_pointer_creator<T>(height * width, managed_memory),
                    vo::cuda::cuda_pointer_deleter<T>
                )
            {};

        };
    }
}

#endif

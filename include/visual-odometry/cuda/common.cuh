#ifndef VO_CUDA_COMMON_H
#define VO_CUDA_COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <stdexcept>
#include <math.h>

#define HANDLE_CUDA_ERROR(err) {vo::cuda::handle_cuda_error(err, __FILE__, __LINE__);}
#define CUDA_BLOCKSIZE 16

namespace vo {
    namespace cuda {

        static const float nan = FP_NAN;

        void handle_cuda_error(int err, const char* file, int line);

        void cuda_malloc_wrapper(void **devPtr, size_t size);
        void cuda_memcpy_to_device_wrapper(void *dst, const void *src, size_t count);
        void cuda_malloc_managed_wrapper(void **devPtr, size_t size, unsigned int flags = 1U);
        void cuda_free_wrapper(void *devPtr);

        void query_devices();

        template<typename T>
        class CudaArray{
            public:
                CudaArray(int height, int width):
                    size_(((size_t) (width * height)) * sizeof(T))
                {
                    vo::cuda::cuda_malloc_wrapper((void**) &raw_gpu_pointer_, size_);
                };

                ~CudaArray() {
                    vo::cuda::cuda_free_wrapper(raw_gpu_pointer_);
                }

                T* data() const {
                    return raw_gpu_pointer_;
                };

                void copyFromHost(const T* host_ptr) {
                    vo::cuda::cuda_memcpy_to_device_wrapper(
                        (void*) raw_gpu_pointer_, (void*) host_ptr, size_
                    );
                }

                inline int size(){
                    return size_;
                };

            private:
                size_t size_;
                T* raw_gpu_pointer_;
        };

        template<typename T>
        class CudaSharedArray{
            public:
                CudaSharedArray():
                    size_(-1),
                    initialized_(false)
                {};

                CudaSharedArray(int height, int width):
                    size_(width * height * sizeof(T))
                {
                    vo::cuda::cuda_malloc_managed_wrapper((void**) &raw_pointer_, size_);
                    initialized_ = true;
                };

                ~CudaSharedArray() {
                    if (initialized_)
                        vo::cuda::cuda_free_wrapper(raw_pointer_);
                };

                void init(int height, int width) {
                    size_ = width * height * sizeof(T);
                    vo::cuda::cuda_malloc_managed_wrapper((void**) &raw_pointer_, size_);
                    initialized_ = true;
                };

                T* data() const {
                    if(!initialized_)
                        throw std::runtime_error("Cannot access pointer of not initialized CUDA Shared Array");

                    return raw_pointer_;
                };

                inline int size(){
                    if(!initialized_)
                        throw std::runtime_error("Cannot access size of not initialized CUDA Shared Array");

                    return size_;
                };

            private:
                int size_;
                T* raw_pointer_;
                bool initialized_;
        };
    }
}

#endif

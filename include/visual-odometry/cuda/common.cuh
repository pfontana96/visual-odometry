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
                    // vo::cuda::cuda_free_wrapper(raw_gpu_pointer_);
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

                CudaSharedArray(int height, int width):
                    size_(width * height * sizeof(T))
                {
                    vo::cuda::cuda_malloc_managed_wrapper((void**) &raw_pointer_, size_);
                };

                ~CudaSharedArray() {
                    vo::cuda::cuda_free_wrapper(raw_pointer_);
                };

                T* data() const {
                    // return raw_pointer_.get();
                    return raw_pointer_;
                };

                inline int size(){
                    return size_;
                };

            private:

                int size_;
                // std::unique_ptr<T, void(*)(T*)> raw_pointer_;
                T* raw_pointer_;

                // static void ptr_deleter_ (T* ptr){
                //     vo::cuda::cuda_free_wrapper(ptr);
                // };

                // T* ptr_creator_(uint32_t size){
                //     T* ptr;
                //     vo::cuda::cuda_malloc_managed_wrapper((void**) &ptr, size * sizeof(T));
                //     return ptr;
                // };

        };
    }
}

#endif

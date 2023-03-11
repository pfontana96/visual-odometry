#include <cuda/common.cuh>

#include <cuda_runtime.h>

namespace vo {
    namespace cuda {

        void cuda_init_device(){
            // NOTE: Not support for multiple GPUs yet
            vo::cuda::query_devices();
            HANDLE_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
            HANDLE_CUDA_ERROR(cudaSetDevice(0));
        }

        void handle_cuda_error(int err, const char* file, int line)
        {
            if(err != cudaSuccess)
            {
                printf("%s in %s at line %d\n", cudaGetErrorString((cudaError_t) err), file, line);
                exit(EXIT_FAILURE);
            }
        };
        
        void query_devices()
        {
            int dev_count;
            cudaGetDeviceCount(&dev_count);
            for(int i = 0; i < dev_count; i++)
            {
                cudaDeviceProp dev_prop;
                cudaGetDeviceProperties(&dev_prop, i);
                printf("Found CUDA Capable device %s (%d.%d)\n", dev_prop.name, 
                                                            dev_prop.major,
                                                            dev_prop.minor);
            }
        }

        void cuda_malloc_wrapper(void **devPtr, size_t size) {
            HANDLE_CUDA_ERROR(cudaMalloc(devPtr, size));
        }

        void cuda_memcpy_to_device_wrapper(void *dst, const void *src, size_t count) {
            HANDLE_CUDA_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }

        void cuda_memcpy_to_host_wrapper(void *dst, const void *src, size_t count) {
            HANDLE_CUDA_ERROR(cudaMemcpy(dst, src, count, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        void cuda_malloc_managed_wrapper(void **devPtr, size_t size, unsigned int flags) {
            HANDLE_CUDA_ERROR(cudaMallocManaged(devPtr, size, flags));
        }

        void cuda_free_wrapper(void *devPtr) {
            HANDLE_CUDA_ERROR(cudaFree(devPtr));
        }

    } // namespace cuda
} // namespace vo
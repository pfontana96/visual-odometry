#ifndef VO_UTILS_COMMONS_H
#define VO_UTILS_COMMONS_H


#include <limits>

#include <utils/types.h>
#ifdef VO_CUDA_ENABLED
#include <cuda/common.cuh>
#endif

namespace vo {
    namespace util {

        inline bool isfinite(float number) {
            bool result = false;
            
            #ifdef VO_CUDA_ENABLED
            result = number != vo::cuda::nan;            

            #else

            result = std::isfinite(number);
            #endif

            return result;
        };

    } // namespace util
} // namespace vo


#endif

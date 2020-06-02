#pragma once

#include "../../base.h"

#include <iostream>

#ifdef USE_RMM
#include <rmm/rmm.h>
#endif

namespace gcurval
{

inline cudaError_t _cudaCall(cudaError_t return_value, const char* file, size_t line)
{
    cudaError_t cudaStatus = return_value;
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "\n\nCUDA ERROR %d - %s\n    in %s : (%zu)\n\n", cudaStatus, cudaGetErrorString(cudaStatus), file, line);
        exit(-1);
    }
    return cudaStatus;
}


#ifdef NDEBUG

#define cc(return_value) return_value

#else

#define cc(return_value) gcurval::_cudaCall(return_value, __FILE__, __LINE__)

#endif // NDEBUG

inline cudaError_t cet()
{
    return cc(cudaPeekAtLastError());
}

#ifdef USE_RMM

#define gMalloc(devPtr, size) ::RMM_ALLOC(devPtr, size, 0)

#define gFree(devPtr) ::RMM_FREE(devPtr, 0)

#define gFree_s(ptr) \
{                    \
    if (ptr) \
    { \
        gFree(ptr); \
        ptr = nullptr; \
    } \
}

#else

#define gMalloc(devPtr, size) ::cudaMalloc(devPtr, size)

#define gFree(devPtr) ::cudaFree(devPtr)

#define gFree_s(ptr) \
{                    \
    if (ptr) \
    { \
        cc(gFree(ptr)); \
        ptr = nullptr; \
    } \
}

#endif // !USE_RMM

}

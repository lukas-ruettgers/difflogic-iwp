#pragma once
#include <torch/extension.h>

//#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>

#include "conv2d.h"

#define BACKWARD_W_BATCH_THREADS 32
#define BACKWARD_W_BATCH_THREADS_LUT 16
#define BACKWARD_W_CONV_BATCH_THREADS 4
// #define BACKWARD_W_CONV2D_BATCH_THREADS 32
#define BACKWARD_W_CONV2D_BATCH_THREADS 32
#define FORWARD_W_CONV2D_BATCH_THREADS 32

#define CONV2D_DIV_THREADS_PER_BLOCK_Y 4


/**********************************************************************************************************************/

// Tensor.type() is deprecated
// #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)


// adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

/**********************************************************************************************************************/

static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) { return atomicAdd(reinterpret_cast<__half *>(address), val); }

static inline __device__ float gpuAtomicAdd(float *address, float val) { return atomicAdd(address, val); }

static inline __device__ double gpuAtomicAdd(double *address, double val) { return atomicAdd(address, val); }

/**********************************************************************************************************************/
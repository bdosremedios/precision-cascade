#include "tools/cuda_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <format>
#include <string>

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::format(
                "cudaError_t {:d}: {} {}",
                static_cast<int>(error),
                cudaGetErrorName(error),
                cudaGetErrorString(error)
            )
        );
    }
}

void check_kernel_launch(
    cudaError_t error,
    std::string function_name,
    std::string kernel_name,
    int n_blocks,
    int n_threads
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::format(
                "cuda kernel {}<<<{}, {}>>> in {} failed with error {} ({}: {})",
                kernel_name,
                n_blocks, n_threads,
                function_name,
                static_cast<int>(error),
                cudaGetErrorName(error),
                cudaGetErrorString(error)
            )
        );
    }
}

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::format("cublasStatus_t failure: {:d}", static_cast<int>(status)));
    }
}

void check_cusparse_status(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::format("cusparseStatus_t failure: {:d}", static_cast<int>(status)));
    }
}
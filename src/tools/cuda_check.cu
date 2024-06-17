#include "tools/cuda_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            "cudaError_t " + std::to_string(static_cast<int>(error)) +
            ": " + cudaGetErrorName(error) + " " + cudaGetErrorString(error)
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
            "cuda kernel " + kernel_name +
            "<<<" + std::to_string(n_blocks) + ", " +
            std::to_string(n_threads) + ">>> in " +
            function_name + " failed with error " +
            std::to_string(static_cast<int>(error)) +
            "(" + cudaGetErrorName(error) + ": " +
            cudaGetErrorString(error) + ")"
        );
    }
}

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cublasStatus_t failure: " +
            std::to_string(static_cast<int>(status))
        );
    }
}

void check_cusparse_status(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cusparseStatus_t failure: " +
            std::to_string(static_cast<int>(status))
        );
    }
}
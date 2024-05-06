#include <stdexcept>
#include <format>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::format("cudaError_t failure: {:d}", static_cast<int>(error)));
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
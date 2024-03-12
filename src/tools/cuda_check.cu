#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error("cudaError_t failure: " + std::to_string(error));
    }
}

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasStatus_t failure: " + std::to_string(status));
    }
}

void check_cusparse_status(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseStatus_t failure: " + std::to_string(status));
    }
}
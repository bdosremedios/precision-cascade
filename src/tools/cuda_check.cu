#include <stdexcept>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasStatus_t failure: " + std::to_string(status));
    }
}

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error("cudaError_t failure: " + std::to_string(error));
    }
}
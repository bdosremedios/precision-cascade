#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK

#include <cuda_runtime.h>
#include <cublas_v2.h>

void check_cublas_status(cublasStatus_t status);

void check_cuda_error(cudaError_t error);

#endif
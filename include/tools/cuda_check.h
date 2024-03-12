#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

void check_cuda_error(cudaError_t error);

void check_cublas_status(cublasStatus_t status);

void check_cusparse_status(cusparseStatus_t status);

#endif
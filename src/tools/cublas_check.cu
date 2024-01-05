#include "tools/cublas_check.h"

#include <stdexcept>
#include <string>

void check_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublas error: " + std::to_string(status));
    }
}
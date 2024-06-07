#include "tools/cuHandleBundle.h"

cublasHandle_t cuHandleBundle::cublas_handle;

cusparseHandle_t cuHandleBundle::cusparse_handle;

void cuHandleBundle::create() {
    check_cublas_status(cublasCreate(&cublas_handle));
    check_cublas_status(
        cublasSetPointerMode(
            cublas_handle,
            CUBLAS_POINTER_MODE_DEVICE
        )
    );
    check_cusparse_status(cusparseCreate(&cusparse_handle));
    check_cusparse_status(
        cusparseSetPointerMode(
            cusparse_handle,
            CUSPARSE_POINTER_MODE_DEVICE
        )
    );
}

void cuHandleBundle::destroy() {
    check_cublas_status(cublasDestroy(cublas_handle));
    check_cusparse_status(cusparseDestroy(cusparse_handle));
}

cublasHandle_t cuHandleBundle::get_cublas_handle() const {
    return cublas_handle;
}

cusparseHandle_t cuHandleBundle::get_cusparse_handle() const {
    return cusparse_handle;
}

bool cuHandleBundle::operator==(const cuHandleBundle &other) const {
    return (
        (cublas_handle == other.cublas_handle) &&
        (cusparse_handle == other.cusparse_handle)
    );
}
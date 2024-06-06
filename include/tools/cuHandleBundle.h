#ifndef CUHANDLEBUNDLE_H
#define CUHANDLEBUNDLE_H

#include "cuda_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

class cuHandleBundle
{
private:

    static cublasHandle_t cublas_handle;
    static cusparseHandle_t cusparse_handle;

public:

    cuHandleBundle() {}

    ~cuHandleBundle() {}

    cuHandleBundle(const cuHandleBundle &) {}
    cuHandleBundle & operator=(const cuHandleBundle &) {
        return *this;
    }

    void create();

    void destroy();

    cublasHandle_t get_cublas_handle() const;

    cusparseHandle_t get_cusparse_handle() const;

    bool operator==(const cuHandleBundle &) const;

};

#endif
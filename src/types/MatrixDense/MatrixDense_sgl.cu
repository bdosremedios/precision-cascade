#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<float> MatrixDense<float>::operator*(const float &scalar) const {

    MatrixDense<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar, CUDA_R_32F, c.d_mat, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;
}
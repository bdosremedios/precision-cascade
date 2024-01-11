#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<__half> MatrixDense<__half>::operator*(const __half &scalar) const {

    MatrixDense<__half> c(*this);

    float scalar_cast = static_cast<float>(scalar);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar_cast, CUDA_R_32F, c.d_mat, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return c;
}
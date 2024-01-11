#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixDense<double> MatrixDense<double>::operator*(const double &scalar) const {

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols, &scalar, CUDA_R_64F, c.d_mat, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;
}
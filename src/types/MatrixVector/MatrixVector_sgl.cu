#include "types/MatrixVector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixVector<float> MatrixVector<float>::operator*(const float &scalar) const {

    MatrixVector<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
MatrixVector<float> & MatrixVector<float>::operator*=(const float &scalar) {

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
MatrixVector<float> MatrixVector<float>::operator+(const MatrixVector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    MatrixVector<float> c(*this);
    float alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
MatrixVector<float> MatrixVector<float>::operator-(const MatrixVector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    MatrixVector<float> c(*this);
    float alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
MatrixVector<float> & MatrixVector<float>::operator+=(const MatrixVector<float> &vec) {

    check_vecvec_op_compatibility(vec);

    float alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
MatrixVector<float> & MatrixVector<float>::operator-=(const MatrixVector<float> &vec) {

    check_vecvec_op_compatibility(vec);

    float alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
float MatrixVector<float>::dot(const MatrixVector<float> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    float result;

    check_cublas_status(
        cublasDotEx(
            handle, m_rows, d_vec, CUDA_R_32F, 1, vec.d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
        )
    );

    return result;

}

template<>
float MatrixVector<float>::norm() const {

    float result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
        )
    );

    return result;

}
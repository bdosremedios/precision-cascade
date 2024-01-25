#include "types/Vector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
Vector<float> Vector<float>::operator*(const float &scalar) const {

    Vector<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
Vector<float> & Vector<float>::operator*=(const float &scalar) {

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
Vector<float> Vector<float>::operator+(const Vector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<float> c(*this);
    float alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
Vector<float> Vector<float>::operator-(const Vector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<float> c(*this);
    float alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_32F, 1, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
Vector<float> & Vector<float>::operator+=(const Vector<float> &vec) {

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
Vector<float> & Vector<float>::operator-=(const Vector<float> &vec) {

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
float Vector<float>::dot(const Vector<float> &vec) const {

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
float Vector<float>::norm() const {

    float result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
        )
    );

    return result;

}
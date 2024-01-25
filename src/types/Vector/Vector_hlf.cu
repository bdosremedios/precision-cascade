#include "types/Vector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
Vector<__half> Vector<__half>::operator*(const __half &scalar) const {

    Vector<__half> c(*this);
    float *scalar_cast = static_cast<float *>(malloc(sizeof(float)));
    *scalar_cast = static_cast<float>(scalar);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, scalar_cast, CUDA_R_32F, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    free(scalar_cast);

    return c;

}

template<>
Vector<__half> & Vector<__half>::operator*=(const __half &scalar) {

    float scalar_cast = static_cast<float>(scalar);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar_cast, CUDA_R_32F, d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );


    return *this;

}

template<>
Vector<__half> Vector<__half>::operator+(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);
    float alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
Vector<__half> Vector<__half>::operator-(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);
    float alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, c.d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return c;

}

template<>
Vector<__half> & Vector<__half>::operator+=(const Vector<__half> &vec) {

    check_vecvec_op_compatibility(vec);

    float alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
Vector<__half> & Vector<__half>::operator-=(const Vector<__half> &vec) {

    check_vecvec_op_compatibility(vec);

    float alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_32F, vec.d_vec, CUDA_R_16F, 1, d_vec, CUDA_R_16F, 1, CUDA_R_32F
        )
    );

    return *this;

}

template<>
__half Vector<__half>::dot(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    __half result;

    check_cublas_status(
        cublasDotEx(
            handle, m_rows, d_vec, CUDA_R_16F, 1, vec.d_vec, CUDA_R_16F, 1, &result, CUDA_R_16F, CUDA_R_32F
        )
    );

    return result;

}

template<>
__half Vector<__half>::norm() const {

    __half result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_16F, 1, &result, CUDA_R_16F, CUDA_R_32F
        )
    );

    return result;

}
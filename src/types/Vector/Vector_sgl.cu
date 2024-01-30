#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/Vector.h"

Vector<float> Vector<float>::operator*(const float &scalar) const {

    Vector<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, c.d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return c;

}

Vector<float> & Vector<float>::operator*=(const float &scalar) {

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_32F, d_vec, CUDA_R_32F, 1, CUDA_R_32F
        )
    );

    return *this;

}

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

float Vector<float>::norm() const {

    float result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_32F, 1, &result, CUDA_R_32F, CUDA_R_32F
        )
    );

    return result;

}

namespace vec_sgl_kern
{
    __global__ void cast_to_half(float *vec_src, half *vec_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            vec_dest[tid] = __float2half(vec_src[tid]);
        }
    }

    __global__ void cast_to_double(float *vec_src, double *vec_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            vec_dest[tid] = static_cast<double>(vec_src[tid]);
        }
    }
}

Vector<__half> Vector<float>::to_half() const {
    
    Vector<__half> created_vec(handle, m_rows);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vec_sgl_kern::cast_to_half<<<NUM_THREADS, NUM_BLOCKS>>>(d_vec, created_vec.d_vec, m_rows);

    return created_vec;

}

Vector<float> Vector<float>::to_float() const { return Vector<float>(*this); }

Vector<double> Vector<float>::to_double() const {
    
    Vector<double> created_vec(handle, m_rows);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vec_sgl_kern::cast_to_double<<<NUM_THREADS, NUM_BLOCKS>>>(d_vec, created_vec.d_vec, m_rows);

    return created_vec;

}
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/Vector.h"

Vector<__half> Vector<__half>::operator*(const Scalar<__half> &scalar) const {

    Vector<__half> c(*this);

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            handle, m_rows,
            temp_cast.d_scalar, CUDA_R_32F,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<__half> & Vector<__half>::operator*=(const Scalar<__half> &scalar) {

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            handle, m_rows,
            temp_cast.d_scalar, CUDA_R_32F,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Vector<__half> Vector<__half>::operator+(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);
    Scalar<float> alpha(1.);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows,
            alpha.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<__half> Vector<__half>::operator-(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);
    Scalar<float> alpha(-1.);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows,
            alpha.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<__half> & Vector<__half>::operator+=(const Vector<__half> &vec) {

    check_vecvec_op_compatibility(vec);

    Scalar<float> alpha(1.);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows,
            alpha.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Vector<__half> & Vector<__half>::operator-=(const Vector<__half> &vec) {

    check_vecvec_op_compatibility(vec);

    Scalar<float> alpha(-1.);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows,
            alpha.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Scalar<__half> Vector<__half>::dot(const Vector<__half> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    Scalar<__half> result;

    check_cublas_status(
        cublasDotEx(
            handle, m_rows,
            d_vec, CUDA_R_16F, 1,
            vec.d_vec, CUDA_R_16F, 1,
            result.d_scalar, CUDA_R_16F,
            CUDA_R_32F
        )
    );

    return result;

}

Scalar<__half> Vector<__half>::norm() const {

    Scalar<__half> result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_16F, 1, result.d_scalar, CUDA_R_16F, CUDA_R_32F
        )
    );

    return result;

}

namespace vec_hlf_kern
{
    __global__ void cast_to_float(__half *scalar_src, float *scalar_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            scalar_dest[tid] = __half2float(scalar_src[tid]);
        }
    }

    __global__ void cast_to_double(__half *scalar_src, double *scalar_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            scalar_dest[tid] = static_cast<double>(scalar_src[tid]);
        }
    }
}

Vector<__half> Vector<__half>::to_half() const { return Vector<__half>(*this); }

Vector<float> Vector<__half>::to_float() const {
    
    Vector<float> created_vec(handle, m_rows);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vec_hlf_kern::cast_to_float<<<NUM_THREADS, NUM_BLOCKS>>>(d_vec, created_vec.d_vec, m_rows);

    return created_vec;

}

Vector<double> Vector<__half>::to_double() const {
    
    Vector<double> created_vec(handle, m_rows);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vec_hlf_kern::cast_to_double<<<NUM_THREADS, NUM_BLOCKS>>>(d_vec, created_vec.d_vec, m_rows);

    return created_vec;

}
#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/Vector/Vector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cascade {

template <>
Vector<__half> Vector<__half>::operator*(
    const Scalar<__half> &scalar
) const {

    Vector<__half> c(*this);

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            temp_cast.d_scalar, CUDA_R_32F,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

template <>
Vector<__half> & Vector<__half>::operator*=(
    const Scalar<__half> &scalar
) {

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            temp_cast.d_scalar, CUDA_R_32F,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

template <>
Vector<__half> Vector<__half>::operator+(
    const Vector<__half> &vec
) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

template <>
Vector<__half> Vector<__half>::operator-(
    const Vector<__half> &vec
) const {

    check_vecvec_op_compatibility(vec);

    Vector<__half> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            c.d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

template <>
Vector<__half> & Vector<__half>::operator+=(
    const Vector<__half> &vec
) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

template <>
Vector<__half> & Vector<__half>::operator-=(
    const Vector<__half> &vec
) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_16F, 1,
            d_vec, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

template <>
Scalar<__half> Vector<__half>::dot(
    const Vector<__half> &vec
) const {

    check_vecvec_op_compatibility(vec);
    
    Scalar<__half> result;

    check_cublas_status(
        cublasDotEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_16F, 1,
            vec.d_vec, CUDA_R_16F, 1,
            result.d_scalar, CUDA_R_16F,
            CUDA_R_32F
        )
    );

    return result;

}

template <>
Scalar<__half> Vector<__half>::norm() const {

    Scalar<__half> result;

    check_cublas_status(
        cublasNrm2Ex(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_16F, 1,
            result.d_scalar, CUDA_R_16F,
            CUDA_R_32F
        )
    );

    return result;

}

template <>
Vector<__half> Vector<__half>::to_half() const {
    return Vector<__half>(*this);
}

template <>
Vector<float> Vector<__half>::to_float() const {
    
    Vector<float> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(m_rows) /
        static_cast<double>(NUM_THREADS)
    );

    if (NUM_BLOCKS > 0) {

        vector_hlf_kernels::cast_to_float<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<__half>::to_float",
            "vector_hlf_kernels::cast_to_float",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}

template <>
Vector<double> Vector<__half>::to_double() const {
    
    Vector<double> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(m_rows) /
        static_cast<double>(NUM_THREADS)
    );

    if (NUM_BLOCKS > 0) {

        vector_hlf_kernels::cast_to_double<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<__half>::to_double",
            "vector_hlf_kernels::cast_to_double",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}

}
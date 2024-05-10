#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

#include "types/Vector/Vector.h"

Vector<__half> Vector<__half>::operator*(const Scalar<__half> &scalar) const {

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

Vector<__half> & Vector<__half>::operator*=(const Scalar<__half> &scalar) {

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

Vector<__half> Vector<__half>::operator+(const Vector<__half> &vec) const {

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

Vector<__half> Vector<__half>::operator-(const Vector<__half> &vec) const {

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

Vector<__half> & Vector<__half>::operator+=(const Vector<__half> &vec) {

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

Vector<__half> & Vector<__half>::operator-=(const Vector<__half> &vec) {

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

Scalar<__half> Vector<__half>::dot(const Vector<__half> &vec) const {

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

Vector<__half> Vector<__half>::to_half() const { return Vector<__half>(*this); }

Vector<float> Vector<__half>::to_float() const {
    
    Vector<float> created_vec(cu_handles, m_rows);

    double NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vector_hlf_kernels::cast_to_float<<<NUM_BLOCKS, NUM_THREADS>>>(d_vec, created_vec.d_vec, m_rows);
    check_kernel_launch(
        cudaGetLastError(),
        "Vector<__half>::to_float",
        "vector_hlf_kernels::cast_to_float",
        NUM_BLOCKS, NUM_THREADS
    );

    return created_vec;

}

Vector<double> Vector<__half>::to_double() const {
    
    Vector<double> created_vec(cu_handles, m_rows);

    double NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );
    vector_hlf_kernels::cast_to_double<<<NUM_BLOCKS, NUM_THREADS>>>(d_vec, created_vec.d_vec, m_rows);
    check_kernel_launch(
        cudaGetLastError(),
        "Vector<__half>::to_double",
        "vector_hlf_kernels::cast_to_double",
        NUM_BLOCKS, NUM_THREADS
    );

    return created_vec;

}
#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/Vector/Vector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

Vector<float> Vector<float>::operator*(const Scalar<float> &scalar) const {

    Vector<float> c(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            scalar.d_scalar, CUDA_R_32F,
            c.d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<float> & Vector<float>::operator*=(const Scalar<float> &scalar) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            scalar.d_scalar, CUDA_R_32F,
            d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Vector<float> Vector<float>::operator+(const Vector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<float> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_32F, 1,
            c.d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<float> Vector<float>::operator-(const Vector<float> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<float> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_32F, 1,
            c.d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return c;

}

Vector<float> & Vector<float>::operator+=(const Vector<float> &vec) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_32F, 1,
            d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Vector<float> & Vector<float>::operator-=(const Vector<float> &vec) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_F.d_scalar, CUDA_R_32F,
            vec.d_vec, CUDA_R_32F, 1,
            d_vec, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

Scalar<float> Vector<float>::dot(const Vector<float> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    Scalar<float> result;

    check_cublas_status(
        cublasDotEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_32F, 1,
            vec.d_vec, CUDA_R_32F, 1,
            result.d_scalar, CUDA_R_32F,
            CUDA_R_32F
        )
    );

    return result;

}

Scalar<float> Vector<float>::norm() const {

    Scalar<float> result;

    check_cublas_status(
        cublasNrm2Ex(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_32F, 1,
            result.d_scalar, CUDA_R_32F,
            CUDA_R_32F
        )
    );

    return result;

}

Vector<__half> Vector<float>::to_half() const {
    
    Vector<__half> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = static_cast<int>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );

    if (NUM_BLOCKS > 0) {

        vector_sgl_kernels::cast_to_half<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<float>::to_half",
            "vector_sgl_kernels::cast_to_half",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}

Vector<float> Vector<float>::to_float() const { return Vector<float>(*this); }

Vector<double> Vector<float>::to_double() const {
    
    Vector<double> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = static_cast<int>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );

    if (NUM_BLOCKS > 0) {

        vector_sgl_kernels::cast_to_double<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<float>::to_double",
            "vector_sgl_kernels::cast_to_double",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}
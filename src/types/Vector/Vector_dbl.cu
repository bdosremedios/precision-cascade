#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

#include "types/Vector/Vector.h"

Vector<double> Vector<double>::operator*(const Scalar<double> &scalar) const {

    Vector<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            scalar.d_scalar, CUDA_R_64F,
            c.d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

Vector<double> & Vector<double>::operator*=(const Scalar<double> &scalar) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            scalar.d_scalar, CUDA_R_64F,
            d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return *this;

}

Vector<double> Vector<double>::operator+(const Vector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_D.d_scalar, CUDA_R_64F,
            vec.d_vec, CUDA_R_64F, 1,
            c.d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

Vector<double> Vector<double>::operator-(const Vector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_D.d_scalar, CUDA_R_64F,
            vec.d_vec, CUDA_R_64F, 1,
            c.d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

Vector<double> & Vector<double>::operator+=(const Vector<double> &vec) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_ONE_D.d_scalar, CUDA_R_64F,
            vec.d_vec, CUDA_R_64F, 1,
            d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return *this;

}

Vector<double> & Vector<double>::operator-=(const Vector<double> &vec) {

    check_vecvec_op_compatibility(vec);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            SCALAR_MINUS_ONE_D.d_scalar, CUDA_R_64F,
            vec.d_vec, CUDA_R_64F, 1,
            d_vec, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return *this;

}

Scalar<double> Vector<double>::dot(const Vector<double> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    Scalar<double> result;

    check_cublas_status(
        cublasDotEx(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_64F, 1,
            vec.d_vec, CUDA_R_64F, 1,
            result.d_scalar, CUDA_R_64F,
            CUDA_R_64F
        )
    );

    return result;

}

Scalar<double> Vector<double>::norm() const {

    Scalar<double> result;

    check_cublas_status(
        cublasNrm2Ex(
            cu_handles.get_cublas_handle(),
            m_rows,
            d_vec, CUDA_R_64F, 1,
            result.d_scalar, CUDA_R_64F,
            CUDA_R_64F
        )
    );

    return result;

}

Vector<__half> Vector<double>::to_half() const {

    Vector<__half> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = static_cast<int>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );

    if (NUM_BLOCKS > 0) {

        vector_dbl_kernels::cast_to_half<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<double>::to_half",
            "vector_dbl_kernels::cast_to_half",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}

Vector<float> Vector<double>::to_float() const {

    Vector<float> created_vec(cu_handles, m_rows);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = static_cast<int>(
        std::ceil(static_cast<double>(m_rows)/static_cast<double>(NUM_THREADS))
    );

    if (NUM_BLOCKS > 0) {

        vector_dbl_kernels::cast_to_float<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_vec, created_vec.d_vec, m_rows
        );

        check_kernel_launch(
            cudaGetLastError(),
            "Vector<double>::to_float",
            "vector_dbl_kernels::cast_to_float",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_vec;

}

Vector<double> Vector<double>::to_double() const { return Vector<double>(*this); }
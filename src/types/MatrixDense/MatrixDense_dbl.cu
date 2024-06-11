#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixDense/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

template <>
MatrixDense<double> MatrixDense<double>::operator*(
    const Scalar<double> &scalar
) const {

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            scalar.d_scalar, CUDA_R_64F,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

template <>
MatrixDense<double> & MatrixDense<double>::operator*=(
    const Scalar<double> &scalar
) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            scalar.d_scalar, CUDA_R_64F,
            d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return *this;

}

template <>
Vector<double> MatrixDense<double>::operator*(
    const Vector<double> &vec
) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod "
            "(operator*(const Vector<double> &vec))"
        );
    }

    Vector<double> c(Vector<double>::Zero(cu_handles, m_rows));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, n_cols,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
Vector<double> MatrixDense<double>::mult_subset_cols(
    int start, int cols, const Vector<double> &vec
) const {

    if ((start < 0) || ((start+cols) > n_cols) || (cols <= 0)) {
        throw std::runtime_error(
            "MatrixDense: invalid column values in mult_subset_cols"
        );
    }
    if (vec.rows() != cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in mult_subset_cols"
        );
    }

    Vector<double> c(Vector<double>::Zero(cu_handles, m_rows));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, cols,
            SCALAR_ONE_D.d_scalar,
            d_mat+start*m_rows, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, n_cols,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
Vector<double> MatrixDense<double>::transpose_prod(
    const Vector<double> &vec
) const {

    if (vec.rows() != m_rows) {
        throw std::runtime_error("MatrixDense: invalid vec in transpose_prod");
    }

    Vector<double> c(Vector<double>::Zero(cu_handles, n_cols));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, m_rows,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, n_cols,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
Vector<double> MatrixDense<double>::transpose_prod_subset_cols(
    int start, int cols, const Vector<double> &vec
) const {

    if ((start < 0) || ((start+cols) > n_cols) || (cols <= 0)) {
        throw std::runtime_error(
            "MatrixDense: invalid column values in transpose_prod_subset_cols"
        );
    }
    if (vec.rows() != m_rows) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in transpose_prod_subset_cols"
        );
    }

    Vector<double> c(Vector<double>::Zero(cu_handles, cols));

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            cols, 1, m_rows,
            SCALAR_ONE_D.d_scalar,
            d_mat+start*m_rows, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, m_rows,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, cols,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator*(
    const MatrixDense<double> &mat
) const {

    if (mat.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix-matrix prod "
            "(operator*(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(
        MatrixDense<double>::Zero(cu_handles, m_rows, mat.cols())
    );

    check_cublas_status(
        cublasGemmEx(
            cu_handles.get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, mat.cols(), n_cols,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            mat.d_mat, CUDA_R_64F, n_cols,
            SCALAR_ZERO_D.d_scalar,
            c.d_mat, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator+(
    const MatrixDense<double> &mat
) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix add "
            "(operator+(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            SCALAR_ONE_D.d_scalar, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

template <>
MatrixDense<double> MatrixDense<double>::operator-(
    const MatrixDense<double> &mat
) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix subtract "
            "(operator-(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            SCALAR_MINUS_ONE_D.d_scalar, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

template <>
Scalar<double> MatrixDense<double>::norm() const {

    Scalar<double> result;

    check_cublas_status(
        cublasNrm2Ex(
            cu_handles.get_cublas_handle(),
            m_rows*n_cols,
            d_mat, CUDA_R_64F, 1,
            result.d_scalar, CUDA_R_64F,
            CUDA_R_64F
        )
    );

    return result;

}

template <>
MatrixDense<__half> MatrixDense<double>::to_half() const {

    MatrixDense<__half> created_mat(cu_handles, m_rows, n_cols);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(m_rows*n_cols) /
        static_cast<double>(NUM_THREADS)
    );

    if (NUM_BLOCKS > 0) {

        generalmatrix_dbl_kernels::cast_to_half<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_mat, created_mat.d_mat, m_rows*n_cols
        );

        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<double>::to_half",
            "generalmatrix_dbl_kernels::cast_to_half",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_mat;

}

template <>
MatrixDense<float> MatrixDense<double>::to_float() const {

    MatrixDense<float> created_mat(cu_handles, m_rows, n_cols);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(m_rows*n_cols) /
        static_cast<double>(NUM_THREADS)
    );

    if (NUM_BLOCKS > 0) {

        generalmatrix_dbl_kernels::cast_to_float<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_mat, created_mat.d_mat, m_rows*n_cols
        );

        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<double>::to_float",
            "generalmatrix_dbl_kernels::cast_to_float",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    return created_mat;

}

template <>
MatrixDense<double> MatrixDense<double>::to_double() const {
    return MatrixDense<double>(*this);
}
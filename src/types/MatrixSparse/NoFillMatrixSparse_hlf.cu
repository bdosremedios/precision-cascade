#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixSparse/NoFillMatrixSparse.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

namespace cascade {

template <>
NoFillMatrixSparse<__half>::NoFillMatrixSparse(
    const MatrixDense<__half> &source_mat
):
    NoFillMatrixSparse(source_mat, CUDA_R_16F)
{}

template <>
NoFillMatrixSparse<__half> NoFillMatrixSparse<__half>::operator*(
    const Scalar<__half> &scalar
) const {

    NoFillMatrixSparse<__half> created_mat(*this);

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            temp_cast.d_scalar, CUDA_R_32F,
            created_mat.d_values, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return created_mat;

}

template <>
NoFillMatrixSparse<__half> & NoFillMatrixSparse<__half>::operator*=(
    const Scalar<__half> &scalar
) {

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            temp_cast.d_scalar, CUDA_R_32F,
            d_values, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

template <>
Vector<__half> NoFillMatrixSparse<__half>::operator*(
    const Vector<__half> &vec
) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "NoFillMatrixSparse: invalid vec in "
            "operator*(const Vector<__half> &vec)"
        );
    }

    Vector<__half> new_vec(cu_handles, m_rows);

    cusparseConstSpMatDescr_t spMatDescr;
    cusparseConstDnVecDescr_t dnVecDescr_orig;
    cusparseDnVecDescr_t dnVecDescr_new;
    
    check_cusparse_status(cusparseCreateConstCsr(
        &spMatDescr,
        m_rows, n_cols, nnz,
        d_row_offsets, d_col_indices, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_16F
    ));
    check_cusparse_status(cusparseCreateConstDnVec(
        &dnVecDescr_orig, n_cols, vec.d_vec, CUDA_R_16F
    ));
    check_cusparse_status(cusparseCreateDnVec(
        &dnVecDescr_new, m_rows, new_vec.d_vec, CUDA_R_16F
    ));

    size_t bufferSize;
    check_cusparse_status(cusparseSpMV_bufferSize(
        cu_handles.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        SCALAR_ONE_F.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_F.d_scalar, dnVecDescr_new,
        CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1,
        &bufferSize
    ));

    float *d_buffer;
    check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

    check_cusparse_status(cusparseSpMV(
        cu_handles.get_cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        SCALAR_ONE_F.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_F.d_scalar, dnVecDescr_new,
        CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1,
        d_buffer
    ));

    check_cuda_error(cudaFree(d_buffer));
    
    check_cusparse_status(cusparseDestroySpMat(spMatDescr));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

    return new_vec;

}

template <>
Vector<__half> NoFillMatrixSparse<__half>::transpose_prod(
    const Vector<__half> &vec
) const {

    if (vec.rows() != m_rows) {
        throw std::runtime_error(
            "NoFillMatrixSparse: invalid vec in "
            "operator*(const Vector<__half> &vec)"
        );
    }

    Vector<__half> new_vec(cu_handles, n_cols);

    cusparseConstSpMatDescr_t spMatDescr;
    cusparseConstDnVecDescr_t dnVecDescr_orig;
    cusparseDnVecDescr_t dnVecDescr_new;
    
    check_cusparse_status(cusparseCreateConstCsr(
        &spMatDescr,
        m_rows, n_cols, nnz,
        d_row_offsets, d_col_indices, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_16F
    ));
    check_cusparse_status(cusparseCreateConstDnVec(
        &dnVecDescr_orig, m_rows, vec.d_vec, CUDA_R_16F
    ));
    check_cusparse_status(cusparseCreateDnVec(
        &dnVecDescr_new, n_cols, new_vec.d_vec, CUDA_R_16F
    ));

    size_t bufferSize;
    check_cusparse_status(cusparseSpMV_bufferSize(
        cu_handles.get_cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        SCALAR_ONE_F.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_F.d_scalar, dnVecDescr_new,
        CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1,
        &bufferSize
    ));

    float *d_buffer;
    check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

    check_cusparse_status(cusparseSpMV(
        cu_handles.get_cusparse_handle(),
        CUSPARSE_OPERATION_TRANSPOSE,
        SCALAR_ONE_F.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_F.d_scalar, dnVecDescr_new,
        CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1,
        d_buffer
    ));

    check_cuda_error(cudaFree(d_buffer));
    
    check_cusparse_status(cusparseDestroySpMat(spMatDescr));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

    return new_vec;

}

template <>
NoFillMatrixSparse<__half> NoFillMatrixSparse<__half>::to_half() const {
    return NoFillMatrixSparse<__half>(*this);
}

template <>
NoFillMatrixSparse<float> NoFillMatrixSparse<__half>::to_float() const {

    NoFillMatrixSparse<float> created_mat(cu_handles, m_rows, n_cols, nnz);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(nnz) /
        static_cast<double>(NUM_THREADS)
    );

    check_cuda_error(cudaMemcpy(
        created_mat.d_row_offsets,
        d_row_offsets,
        mem_size_row_offsets(),
        cudaMemcpyDeviceToDevice
    ));

    check_cuda_error(cudaMemcpy(
        created_mat.d_col_indices,
        d_col_indices,
        mem_size_col_indices(),
        cudaMemcpyDeviceToDevice
    ));

    if (NUM_BLOCKS > 0) {

        generalmatrix_hlf_kernels::cast_to_float<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_values, created_mat.d_values, nnz
        );

        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<__half>::to_float",
            "generalmatrix_hlf_kernels::cast_to_float",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    created_mat.deep_copy_trsv_preprocess(*this);

    return created_mat;

}

template <>
NoFillMatrixSparse<double> NoFillMatrixSparse<__half>::to_double() const {
    
    NoFillMatrixSparse<double> created_mat(cu_handles, m_rows, n_cols, nnz);

    int NUM_THREADS = genmat_gpu_const::MAXTHREADSPERBLOCK;
    int NUM_BLOCKS = std::ceil(
        static_cast<double>(nnz) /
        static_cast<double>(NUM_THREADS)
    );

    check_cuda_error(cudaMemcpy(
        created_mat.d_row_offsets,
        d_row_offsets,
        mem_size_row_offsets(),
        cudaMemcpyDeviceToDevice
    ));

    check_cuda_error(cudaMemcpy(
        created_mat.d_col_indices,
        d_col_indices,
        mem_size_col_indices(),
        cudaMemcpyDeviceToDevice
    ));

    if (NUM_BLOCKS > 0) {

        generalmatrix_hlf_kernels::cast_to_double<<<NUM_BLOCKS, NUM_THREADS>>>(
            d_values, created_mat.d_values, nnz
        );

        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<__half>::to_double",
            "generalmatrix_hlf_kernels::cast_to_double",
            NUM_BLOCKS, NUM_THREADS
        );

    }

    created_mat.deep_copy_trsv_preprocess(*this);

    return created_mat;

}

}
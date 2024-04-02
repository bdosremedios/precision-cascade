#include "types/MatrixSparse/NoFillMatrixSparse.h"

NoFillMatrixSparse<double> NoFillMatrixSparse<double>::operator*(const Scalar<double> &scalar) const {

    NoFillMatrixSparse<double> created_mat(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            scalar.d_scalar, CUDA_R_64F,
            created_mat.d_vals, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return created_mat;

}

NoFillMatrixSparse<double> & NoFillMatrixSparse<double>::operator*=(const Scalar<double> &scalar) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            scalar.d_scalar, CUDA_R_64F,
            d_vals, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return *this;

}

Vector<double> NoFillMatrixSparse<double>::matvec_prod_subroutine(
    const Vector<double> &vec, cusparseOperation_t op
) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "NoFillMatrixSparse: invalid vec in matrix-vector prod (operator*(const Vector<double> &vec))"
        );
    }

    Vector<double> new_vec(cu_handles, m_rows);

    cusparseConstSpMatDescr_t spMatDescr;
    cusparseConstDnVecDescr_t dnVecDescr_orig;
    cusparseDnVecDescr_t dnVecDescr_new;
    
    check_cusparse_status(cusparseCreateConstCsc(
        &spMatDescr,
        m_rows, n_cols, nnz,
        d_col_offsets, d_row_indices, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ));
    check_cusparse_status(cusparseCreateConstDnVec(&dnVecDescr_orig, n_cols, vec.d_vec, CUDA_R_64F));
    check_cusparse_status(cusparseCreateDnVec(&dnVecDescr_new, m_rows, new_vec.d_vec, CUDA_R_64F));

    size_t bufferSize;
    check_cusparse_status(cusparseSpMV_bufferSize(
        cu_handles.get_cusparse_handle(),
        op,
        SCALAR_ONE_D.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_D.d_scalar, dnVecDescr_new,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1,
        &bufferSize
    ));

    double *d_buffer;
    check_cuda_error(cudaMalloc(&d_buffer, bufferSize));

    check_cusparse_status(cusparseSpMV(
        cu_handles.get_cusparse_handle(),
        op,
        SCALAR_ONE_D.d_scalar, spMatDescr, dnVecDescr_orig,
        SCALAR_ZERO_D.d_scalar, dnVecDescr_new,
        CUDA_R_64F,
        CUSPARSE_SPMV_CSR_ALG1,
        d_buffer
    ));

    check_cuda_error(cudaFree(d_buffer));
    
    check_cusparse_status(cusparseDestroySpMat(spMatDescr));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_orig));
    check_cusparse_status(cusparseDestroyDnVec(dnVecDescr_new));

    return new_vec;

}

Vector<double> NoFillMatrixSparse<double>::operator*(const Vector<double> &vec) const {
    return matvec_prod_subroutine(vec, CUSPARSE_OPERATION_NON_TRANSPOSE);
}

NoFillMatrixSparse<__half> NoFillMatrixSparse<double>::to_half() const {

    NoFillMatrixSparse<__half> created_mat(cu_handles, m_rows, n_cols, nnz);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(nnz)/static_cast<double>(NUM_THREADS))
    );

    check_cuda_error(cudaMemcpy(
        created_mat.d_col_offsets,
        d_col_offsets,
        mem_size_col_offsets(),
        cudaMemcpyDeviceToDevice
    ));

    check_cuda_error(cudaMemcpy(
        created_mat.d_row_indices,
        d_row_indices,
        mem_size_row_indices(),
        cudaMemcpyDeviceToDevice
    ));

    generalmatrix_dbl_kernels::cast_to_half<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, nnz
    );

    return created_mat;

}

NoFillMatrixSparse<float> NoFillMatrixSparse<double>::to_float() const {

    NoFillMatrixSparse<float> created_mat(cu_handles, m_rows, n_cols, nnz);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(nnz)/static_cast<double>(NUM_THREADS))
    );

    check_cuda_error(cudaMemcpy(
        created_mat.d_col_offsets,
        d_col_offsets,
        mem_size_col_offsets(),
        cudaMemcpyDeviceToDevice
    ));

    check_cuda_error(cudaMemcpy(
        created_mat.d_row_indices,
        d_row_indices,
        mem_size_row_indices(),
        cudaMemcpyDeviceToDevice
    ));

    generalmatrix_dbl_kernels::cast_to_float<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, nnz
    );

    return created_mat;

}

NoFillMatrixSparse<double> NoFillMatrixSparse<double>::to_double() const {
    return NoFillMatrixSparse<double>(*this);
}
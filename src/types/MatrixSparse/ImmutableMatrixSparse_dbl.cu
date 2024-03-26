#include "types/MatrixSparse/ImmutableMatrixSparse.h"

ImmutableMatrixSparse<double> ImmutableMatrixSparse<double>::operator*(const Scalar<double> &scalar) const {

    ImmutableMatrixSparse<double> created_mat(*this);

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

ImmutableMatrixSparse<double> & ImmutableMatrixSparse<double>::operator*=(const Scalar<double> &scalar) {

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

ImmutableMatrixSparse<__half> ImmutableMatrixSparse<double>::to_half() const {

    ImmutableMatrixSparse<__half> created_mat(cu_handles, m_rows, n_cols, nnz);

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

ImmutableMatrixSparse<float> ImmutableMatrixSparse<double>::to_float() const {

    ImmutableMatrixSparse<float> created_mat(cu_handles, m_rows, n_cols, nnz);

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

ImmutableMatrixSparse<double> ImmutableMatrixSparse<double>::to_double() const {
    return ImmutableMatrixSparse<double>(*this);
}
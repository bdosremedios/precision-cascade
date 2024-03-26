#include "types/MatrixSparse/ImmutableMatrixSparse.h"

ImmutableMatrixSparse<__half> ImmutableMatrixSparse<__half>::operator*(const Scalar<__half> &scalar) const {

    ImmutableMatrixSparse<__half> created_mat(*this);

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            temp_cast.d_scalar, CUDA_R_32F,
            created_mat.d_vals, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return created_mat;

}


ImmutableMatrixSparse<__half> & ImmutableMatrixSparse<__half>::operator*=(const Scalar<__half> &scalar) {

    Scalar<float> temp_cast(scalar.cast<float>());

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            temp_cast.d_scalar, CUDA_R_32F,
            d_vals, CUDA_R_16F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

ImmutableMatrixSparse<__half> ImmutableMatrixSparse<__half>::to_half() const {
    return ImmutableMatrixSparse<__half>(*this);
}

ImmutableMatrixSparse<float> ImmutableMatrixSparse<__half>::to_float() const {

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

    generalmatrix_hlf_kernels::cast_to_float<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, nnz
    );

    return created_mat;

}

ImmutableMatrixSparse<double> ImmutableMatrixSparse<__half>::to_double() const {
    
    ImmutableMatrixSparse<double> created_mat(cu_handles, m_rows, n_cols, nnz);

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

    generalmatrix_hlf_kernels::cast_to_double<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, m_rows*n_cols
    );

    return created_mat;

}
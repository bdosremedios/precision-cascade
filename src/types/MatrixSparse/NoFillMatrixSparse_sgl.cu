#include "types/MatrixSparse/NoFillMatrixSparse.h"

NoFillMatrixSparse<float> NoFillMatrixSparse<float>::operator*(const Scalar<float> &scalar) const {

    NoFillMatrixSparse<float> created_mat(*this);

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            scalar.d_scalar, CUDA_R_32F,
            created_mat.d_vals, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return created_mat;

}

NoFillMatrixSparse<float> & NoFillMatrixSparse<float>::operator*=(const Scalar<float> &scalar) {

    check_cublas_status(
        cublasScalEx(
            cu_handles.get_cublas_handle(),
            nnz,
            scalar.d_scalar, CUDA_R_32F,
            d_vals, CUDA_R_32F, 1,
            CUDA_R_32F
        )
    );

    return *this;

}

NoFillMatrixSparse<__half> NoFillMatrixSparse<float>::to_half() const {

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

    generalmatrix_sgl_kernels::cast_to_half<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, nnz
    );

    return created_mat;

}

NoFillMatrixSparse<float> NoFillMatrixSparse<float>::to_float() const {
    return NoFillMatrixSparse<float>(*this);
}

NoFillMatrixSparse<double> NoFillMatrixSparse<float>::to_double() const {

    NoFillMatrixSparse<double> created_mat(cu_handles, m_rows, n_cols, nnz);

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

    generalmatrix_sgl_kernels::cast_to_double<<<NUM_THREADS, NUM_BLOCKS>>>(
        d_vals, created_mat.d_vals, nnz
    );

    return created_mat;

}
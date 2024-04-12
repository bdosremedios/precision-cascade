#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

#include "types/MatrixSparse/NoFillMatrixSparse.h"

template <typename T>
Vector<T> NoFillMatrixSparse<T>::frwd_sub(Vector<T> &arg_rhs) const {

    Vector<T> soln(arg_rhs);

    int32_t *h_col_offsets = static_cast<int32_t *>(malloc(mem_size_col_offsets()));

    check_cuda_error(cudaMemcpy(
        h_col_offsets, d_col_offsets, mem_size_col_offsets(), cudaMemcpyDeviceToHost
    ));

    for (int i=0; i<n_cols; ++i) {

        // Update solution with column pivot
        nofillmatrixsparse_kernels::update_pivot<T><<<1, 1>>>(
            h_col_offsets[i], d_row_indices, d_vals, soln.d_vec
        );

        // Update solution corresponding to remainder to column
        int remaining_col_size = h_col_offsets[i+1]-h_col_offsets[i];
        if (remaining_col_size > 0) {

            int NBLOCKS = std::ceil(
                static_cast<double>(remaining_col_size)/
                static_cast<double>(genmat_gpu_const::MAXTHREADSPERBLOCK)
            );

            nofillmatrixsparse_kernels::lowtri_update_remaining_col
                <T>
                <<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>
            (
                h_col_offsets[i], h_col_offsets[i+1], d_row_indices, d_vals, soln.d_vec
            );

        }

    }

    free(h_col_offsets);

    return soln;

}

template Vector<__half> NoFillMatrixSparse<__half>::frwd_sub(Vector<__half> &) const;
template Vector<float> NoFillMatrixSparse<float>::frwd_sub(Vector<float> &) const;
template Vector<double> NoFillMatrixSparse<double>::frwd_sub(Vector<double> &) const;
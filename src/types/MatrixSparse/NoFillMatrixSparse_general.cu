#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixSparse/NoFillMatrixSparse.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

namespace cascade {

template <typename TPrecision>
Vector<TPrecision> NoFillMatrixSparse<TPrecision>::back_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    Vector<TPrecision> soln(arg_rhs);

    int *h_row_offsets = static_cast<int *>(malloc(mem_size_row_offsets()));

    check_cuda_error(cudaMemcpy(
        h_row_offsets,
        d_row_offsets,
        mem_size_row_offsets(),
        cudaMemcpyDeviceToHost
    ));

    for (int i=m_rows-1; i>=0; --i) {

        int row_elem_count = h_row_offsets[i+1] - h_row_offsets[i];
        if (row_elem_count > 1) {

            int NBLOCKS = std::ceil(
                static_cast<double>(row_elem_count-1) /
                static_cast<double>(genmat_gpu_const::MAXTHREADSPERBLOCK)
            );
            
            nofillmatrixsparse_kernels::upptri_update_right_of_pivot
                <TPrecision>
                <<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>
            (
                i, h_row_offsets[i], row_elem_count,
                d_col_indices, d_values, soln.d_vec
            );
            check_kernel_launch(
                cudaGetLastError(),
                "NoFillMatrixSparse<TPrecision>::back_sub",
                "nofillmatrixsparse_kernels::upptri_update_right_of_pivot"
                "<TPrecision>"
                "<<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>",
                NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK
            );

        }

        // Update solution with row pivot
        nofillmatrixsparse_kernels::update_row_pivot<TPrecision><<<1, 1>>>(
            i, h_row_offsets[i], d_values, soln.d_vec
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::back_sub",
            "nofillmatrixsparse_kernels::update_row_pivot<TPrecision>",
            1, 1
        );

    }

    free(h_row_offsets);

    return soln;

}

template Vector<__half> NoFillMatrixSparse<__half>::back_sub(
    const Vector<__half> &
) const;
template Vector<float> NoFillMatrixSparse<float>::back_sub(
    const Vector<float> &
) const;
template Vector<double> NoFillMatrixSparse<double>::back_sub(
    const Vector<double> &
) const;

template <typename TPrecision>
Vector<TPrecision> NoFillMatrixSparse<TPrecision>::frwd_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    Vector<TPrecision> soln(arg_rhs);

    int *h_row_offsets = static_cast<int *>(malloc(mem_size_row_offsets()));

    check_cuda_error(cudaMemcpy(
        h_row_offsets,
        d_row_offsets,
        mem_size_row_offsets(),
        cudaMemcpyDeviceToHost
    ));

    for (int i=0; i<m_rows; ++i) {

        int row_elem_count = h_row_offsets[i+1] - h_row_offsets[i];
        if (row_elem_count > 1) {

            int NBLOCKS = std::ceil(
                static_cast<double>(row_elem_count-1) /
                static_cast<double>(genmat_gpu_const::MAXTHREADSPERBLOCK)
            );
            
            nofillmatrixsparse_kernels::lowtri_update_left_of_pivot
                <TPrecision>
                <<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>
            (
                i, h_row_offsets[i], row_elem_count,
                d_col_indices, d_values, soln.d_vec
            );
            check_kernel_launch(
                cudaGetLastError(),
                "NoFillMatrixSparse<TPrecision>::back_sub",
                "nofillmatrixsparse_kernels::upptri_update_right_of_pivot"
                "<TPrecision>"
                "<<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>",
                NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK
            );

        }

        // Update solution with row pivot
        nofillmatrixsparse_kernels::update_row_pivot<TPrecision><<<1, 1>>>(
            i, h_row_offsets[i+1]-1, d_values, soln.d_vec
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::back_sub",
            "nofillmatrixsparse_kernels::update_row_pivot<TPrecision>",
            1, 1
        );

    }

    free(h_row_offsets);

    return soln;

}

template Vector<__half> NoFillMatrixSparse<__half>::frwd_sub(
    const Vector<__half> &
) const;
template Vector<float> NoFillMatrixSparse<float>::frwd_sub(
    const Vector<float> &
) const;
template Vector<double> NoFillMatrixSparse<double>::frwd_sub(
    const Vector<double> &
) const;

}
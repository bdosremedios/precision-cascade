#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixSparse/NoFillMatrixSparse.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

namespace cascade {

// template <typename TPrecision>
// Vector<TPrecision> NoFillMatrixSparse<TPrecision>::back_sub(
//     const Vector<TPrecision> &arg_rhs
// ) const {

//     Vector<TPrecision> soln(arg_rhs);

//     int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));

//     check_cuda_error(cudaMemcpy(
//         h_col_offsets,
//         d_col_offsets,
//         mem_size_col_offsets(),
//         cudaMemcpyDeviceToHost
//     ));

//     for (int j=n_cols-1; j>=0; --j) {

//         // Update solution with column pivot
//         nofillmatrixsparse_kernels::update_pivot<TPrecision><<<1, 1>>>(
//             h_col_offsets[j+1]-1, d_row_indices, d_vals, soln.d_vec
//         );
//         check_kernel_launch(
//             cudaGetLastError(),
//             "NoFillMatrixSparse<TPrecision>::back_sub",
//             "nofillmatrixsparse_kernels::update_pivot<TPrecision>",
//             1, 1
//         );

//         // Update solution corresponding to remainder to column
//         int col_size = h_col_offsets[j+1]-h_col_offsets[j];
//         if (col_size > 1) {

//             int NBLOCKS = std::ceil(
//                 static_cast<double>(col_size-1) /
//                 static_cast<double>(genmat_gpu_const::MAXTHREADSPERBLOCK)
//             );

//             nofillmatrixsparse_kernels::upptri_update_remaining_col
//                 <TPrecision>
//                 <<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>
//             (
//                 h_col_offsets[j], col_size, d_row_indices, d_vals, soln.d_vec
//             );
//             check_kernel_launch(
//                 cudaGetLastError(),
//                 "NoFillMatrixSparse<TPrecision>::back_sub",
//                 "nofillmatrixsparse_kernels::upptri_update_remaining_col"
//                 "<TPrecision>",
//                 NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK
//             );

//         }

//     }

//     free(h_col_offsets);

//     return soln;

// }

// template Vector<__half> NoFillMatrixSparse<__half>::back_sub(
//     const Vector<__half> &
// ) const;
// template Vector<float> NoFillMatrixSparse<float>::back_sub(
//     const Vector<float> &
// ) const;
// template Vector<double> NoFillMatrixSparse<double>::back_sub(
//     const Vector<double> &
// ) const;

// template <typename TPrecision>
// Vector<TPrecision> NoFillMatrixSparse<TPrecision>::frwd_sub(
//     const Vector<TPrecision> &arg_rhs
// ) const {

//     Vector<TPrecision> soln(arg_rhs);

//     int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));

//     check_cuda_error(cudaMemcpy(
//         h_col_offsets,
//         d_col_offsets,
//         mem_size_col_offsets(),
//         cudaMemcpyDeviceToHost
//     ));

//     for (int j=0; j<n_cols; ++j) {

//         // Update solution with column pivot
//         nofillmatrixsparse_kernels::update_pivot<TPrecision><<<1, 1>>>(
//             h_col_offsets[j], d_row_indices, d_vals, soln.d_vec
//         );
//         check_kernel_launch(
//             cudaGetLastError(),
//             "NoFillMatrixSparse<TPrecision>::frwd_sub",
//             "nofillmatrixsparse_kernels::update_pivot<TPrecision>",
//             1, 1
//         );

//         // Update solution corresponding to remainder to column
//         int col_size = h_col_offsets[j+1]-h_col_offsets[j];
//         if (col_size > 1) {

//             int NBLOCKS = std::ceil(
//                 static_cast<double>(col_size-1) /
//                 static_cast<double>(genmat_gpu_const::MAXTHREADSPERBLOCK)
//             );

//             nofillmatrixsparse_kernels::lowtri_update_remaining_col
//                 <TPrecision>
//                 <<<NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK>>>
//             (
//                 h_col_offsets[j], col_size, d_row_indices, d_vals, soln.d_vec
//             );
//             check_kernel_launch(
//                 cudaGetLastError(),
//                 "NoFillMatrixSparse<TPrecision>::frwd_sub",
//                 "nofillmatrixsparse_kernels::lowtri_update_remaining_col"
//                 "<TPrecision>",
//                 NBLOCKS, genmat_gpu_const::MAXTHREADSPERBLOCK
//             );

//         }

//     }

//     free(h_col_offsets);

//     return soln;

// }

// template Vector<__half> NoFillMatrixSparse<__half>::frwd_sub(
//     const Vector<__half> &
// ) const;
// template Vector<float> NoFillMatrixSparse<float>::frwd_sub(
//     const Vector<float> &
// ) const;
// template Vector<double> NoFillMatrixSparse<double>::frwd_sub(
//     const Vector<double> &
// ) const;

}
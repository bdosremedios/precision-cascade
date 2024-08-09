#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixSparse/NoFillMatrixSparse.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

namespace cascade {

template <typename TPrecision>
void NoFillMatrixSparse<TPrecision>::preprocess_trsv(bool is_upptri) {

    if (!trsv_level_set_cnt.empty() || !trsv_level_set_ptrs.empty()) {
        throw std::runtime_error(
            "NoFillMatrixSparse<TPrecision>::preprocess_trsv encountered non-"
            "empty"
        );
    }

    std::vector<int> row_level_set(m_rows, -1);
    std::vector<std::vector<int>> trsv_level_sets;

    int *h_row_offsets = static_cast<int *>(
        malloc(mem_size_row_offsets())
    );
    int *h_col_indices = static_cast<int *>(
        malloc(mem_size_col_indices())
    );

    check_cuda_error(cudaMemcpy(
        h_row_offsets,
        d_row_offsets,
        mem_size_row_offsets(),
        cudaMemcpyDeviceToHost
    ));
    check_cuda_error(cudaMemcpy(
        h_col_indices,
        d_col_indices,
        mem_size_col_indices(),
        cudaMemcpyDeviceToHost
    ));

    int start = 0;
    int end = m_rows;
    int iter = 1;
    if (is_upptri) {
        start = m_rows-1;
        end = -1;
        iter = -1;
    }

    for (int i=start; i != end; i += iter) {

        int max_relying_level_set = -1;
        for (
            int offset = h_row_offsets[i];
            offset < h_row_offsets[i+1];
            ++offset
        ) {
            int j = h_col_indices[offset];
            if (max_relying_level_set < row_level_set[j]) {
                max_relying_level_set = row_level_set[j];
            }
        }

        int level_set = max_relying_level_set + 1;
        row_level_set[i] = level_set;
        if (trsv_level_sets.size() == level_set) {
            trsv_level_sets.push_back(std::vector<int>({i}));
        } else {
            trsv_level_sets[level_set].push_back(i);
        }

    }

    free(h_row_offsets);
    free(h_col_indices);
    
    trsv_level_set_cnt.resize(trsv_level_sets.size());
    trsv_level_set_ptrs.resize(trsv_level_sets.size());
    for (int k=0; k < trsv_level_sets.size(); ++k) {

        trsv_level_set_cnt[k] = trsv_level_sets[k].size();

        int *d_lvl_set_ptr = nullptr;
        int *h_lvl_set_ptr = &(trsv_level_sets[k][0]);

        check_cuda_error(cudaMalloc(
            &d_lvl_set_ptr, trsv_level_set_cnt[k]*sizeof(int)
        ));
        trsv_level_set_ptrs[k] = d_lvl_set_ptr;

        check_cuda_error(cudaMemcpy(
            trsv_level_set_ptrs[k],
            h_lvl_set_ptr,
            trsv_level_set_cnt[k]*sizeof(int),
            cudaMemcpyHostToDevice
        ));

    }

}

template void NoFillMatrixSparse<__half>::preprocess_trsv(bool);
template void NoFillMatrixSparse<float>::preprocess_trsv(bool);
template void NoFillMatrixSparse<double>::preprocess_trsv(bool);

template <typename TPrecision>
void NoFillMatrixSparse<TPrecision>::slow_back_sub(
    Vector<TPrecision> &soln_rhs
) const {

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
                d_col_indices, d_values, soln_rhs.d_vec
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
            i, h_row_offsets[i], d_values, soln_rhs.d_vec
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::back_sub",
            "nofillmatrixsparse_kernels::update_row_pivot<TPrecision>",
            1, 1
        );

    }

    free(h_row_offsets);

}

template void NoFillMatrixSparse<__half>::slow_back_sub(
    Vector<__half> &
) const;
template void NoFillMatrixSparse<float>::slow_back_sub(
    Vector<float> &
) const;
template void NoFillMatrixSparse<double>::slow_back_sub(
    Vector<double> &
) const;

// Level set parallel version of back sub with 1 warp assigned to each component
// in each level set, made to be robust against nvidia archicteture changes
// since those could effect sync free algorithms
template <typename TPrecision>
void NoFillMatrixSparse<TPrecision>::fast_back_sub(
    Vector<TPrecision> &soln_rhs
) const {

    for (int k=0; k < trsv_level_set_cnt.size(); ++k) {

        int lvl_set_size = trsv_level_set_cnt[k];
        nofillmatrixsparse_kernels::fast_back_sub_solve_level_set
            <TPrecision>
            <<<lvl_set_size, genmat_gpu_const::WARPSIZE>>>
        (
            trsv_level_set_ptrs[k], soln_rhs.d_vec,
            d_row_offsets, d_col_indices, d_values
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::fast_back_sub",
            "nofillmatrixsparse_kernels::fast_back_sub_solve_level_set"
            "<TPrecision><<<lvl_set_size, genmat_gpu_const::WARPSIZE>>>",
            lvl_set_size, genmat_gpu_const::WARPSIZE
        );

    }

}

template void NoFillMatrixSparse<__half>::fast_back_sub(
    Vector<__half> &
) const;
template void NoFillMatrixSparse<float>::fast_back_sub(
    Vector<float> &
) const;
template void NoFillMatrixSparse<double>::fast_back_sub(
    Vector<double> &
) const;

template <typename TPrecision>
Vector<TPrecision> NoFillMatrixSparse<TPrecision>::back_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    check_trsv_dims(arg_rhs);

    Vector<TPrecision> soln(arg_rhs);

    if (get_has_fast_trsv()) {
        fast_back_sub(soln);
    } else {
        slow_back_sub(soln);
    }

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
void NoFillMatrixSparse<TPrecision>::slow_frwd_sub(
    Vector<TPrecision> &soln_rhs
) const {

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
                d_col_indices, d_values, soln_rhs.d_vec
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
            i, h_row_offsets[i+1]-1, d_values, soln_rhs.d_vec
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::back_sub",
            "nofillmatrixsparse_kernels::update_row_pivot<TPrecision>",
            1, 1
        );

    }

    free(h_row_offsets);

}

template void NoFillMatrixSparse<__half>::slow_frwd_sub(
    Vector<__half> &
) const;
template void NoFillMatrixSparse<float>::slow_frwd_sub(
    Vector<float> &
) const;
template void NoFillMatrixSparse<double>::slow_frwd_sub(
    Vector<double> &
) const;

// Level set parallel version of frwd sub with 1 warp assigned to each component
// in each level set, made to be robust against nvidia archicteture changes
// since those could effect sync free algorithms
template <typename TPrecision>
void NoFillMatrixSparse<TPrecision>::fast_frwd_sub(
    Vector<TPrecision> &soln_rhs
) const {

    for (int k=0; k < trsv_level_set_cnt.size(); ++k) {

        int lvl_set_size = trsv_level_set_cnt[k];
        nofillmatrixsparse_kernels::fast_frwd_sub_solve_level_set
            <TPrecision>
            <<<lvl_set_size, genmat_gpu_const::WARPSIZE>>>
        (
            trsv_level_set_ptrs[k], soln_rhs.d_vec,
            d_row_offsets, d_col_indices, d_values
        );
        check_kernel_launch(
            cudaGetLastError(),
            "NoFillMatrixSparse<TPrecision>::fast_frwd_sub",
            "nofillmatrixsparse_kernels::fast_frwd_sub_solve_level_set"
            "<TPrecision><<<lvl_set_size, genmat_gpu_const::WARPSIZE>>>",
            lvl_set_size, genmat_gpu_const::WARPSIZE
        );

    }

}

template void NoFillMatrixSparse<__half>::fast_frwd_sub(
    Vector<__half> &
) const;
template void NoFillMatrixSparse<float>::fast_frwd_sub(
    Vector<float> &
) const;
template void NoFillMatrixSparse<double>::fast_frwd_sub(
    Vector<double> &
) const;

template <typename TPrecision>
Vector<TPrecision> NoFillMatrixSparse<TPrecision>::frwd_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    check_trsv_dims(arg_rhs);

    Vector<TPrecision> soln(arg_rhs);

    if (get_has_fast_trsv()) {
        fast_frwd_sub(soln);
    } else {
        slow_frwd_sub(soln);
    }

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
#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/MatrixDense/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace cascade {

template <typename TPrecision>
Vector<TPrecision> MatrixDense<TPrecision>::back_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    if (m_rows != n_cols) {
        throw std::runtime_error(
            "MatrixDense::back_sub: non-square matrix"
        );
    }
    if (m_rows != arg_rhs.rows()) {
        throw std::runtime_error(
            "MatrixDense::back_sub: incompatible matrix and rhs"
        );
    }

    Vector<TPrecision> soln(arg_rhs);

    TPrecision *d_soln = soln.d_vec;
    
    int n_blk = std::ceil(
        static_cast<float>(m_rows) /
        genmat_gpu_const::WARPSIZE
    );

    for (int i=n_blk-1; i>=0; --i) {

        matrixdense_kernels::upptri_blk_solve_warp<TPrecision>
            <<<1, genmat_gpu_const::WARPSIZE>>>
        (
            d_mat, m_rows, i*genmat_gpu_const::WARPSIZE, d_soln
        );
        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<TPrecision>::back_sub",
            "matrixdense_kernels::upptri_blk_solve_warp",
            1, genmat_gpu_const::WARPSIZE
        );

        if (i > 0) {
            matrixdense_kernels::upptri_rect_update_warp
                <TPrecision>
                <<<i, genmat_gpu_const::WARPSIZE>>>
            (
                d_mat, m_rows, i*genmat_gpu_const::WARPSIZE, d_soln
            );
            check_kernel_launch(
                cudaGetLastError(),
                "MatrixDense<TPrecision>::back_sub",
                "matrixdense_kernels::upptri_rect_update_warp",
                i, genmat_gpu_const::WARPSIZE
            );
        } 

    }

    return soln;

}

template Vector<__half> MatrixDense<__half>::back_sub(
    const Vector<__half> &arg_rhs
) const;
template Vector<float> MatrixDense<float>::back_sub(
    const Vector<float> &arg_rhs
) const;
template Vector<double> MatrixDense<double>::back_sub(
    const Vector<double> &arg_rhs
) const;

template <typename TPrecision>
Vector<TPrecision> MatrixDense<TPrecision>::frwd_sub(
    const Vector<TPrecision> &arg_rhs
) const {

    if (m_rows != n_cols) {
        throw std::runtime_error(
            "MatrixDense::frwd_sub: non-square matrix"
        );
    }
    if (m_rows != arg_rhs.rows()) {
        throw std::runtime_error(
            "MatrixDense::frwd_sub: incompatible matrix and rhs"
        );
    }

    Vector<TPrecision> soln(arg_rhs);

    TPrecision *d_soln = soln.d_vec;

    int n_blk = std::ceil(
        static_cast<float>(m_rows) /
        genmat_gpu_const::WARPSIZE
    );

    for (int i=0; i<n_blk; ++i) {

        matrixdense_kernels::lowtri_blk_solve_warp
            <TPrecision>
            <<<1, genmat_gpu_const::WARPSIZE>>>
        (
            d_mat, m_rows, i*genmat_gpu_const::WARPSIZE, d_soln
        );
        check_kernel_launch(
            cudaGetLastError(),
            "MatrixDense<TPrecision>::frwd_sub",
            "matrixdense_kernels::lowtri_blk_solve_warp",
            1, genmat_gpu_const::WARPSIZE
        );

        if (n_blk-1-i > 0) {
            matrixdense_kernels::lowtri_rect_update_warp
                <TPrecision>
                <<<n_blk-1-i, genmat_gpu_const::WARPSIZE>>>
            (
                d_mat, m_rows, i*genmat_gpu_const::WARPSIZE, d_soln
            );
            check_kernel_launch(
                cudaGetLastError(),
                "MatrixDense<TPrecision>::frwd_sub",
                "matrixdense_kernels::lowtri_rect_update_warp",
                n_blk-1-i, genmat_gpu_const::WARPSIZE
            );
        }

    }

    return soln;

}

template Vector<__half> MatrixDense<__half>::frwd_sub(
    const Vector<__half> &arg_rhs
) const;
template Vector<float> MatrixDense<float>::frwd_sub(
    const Vector<float> &arg_rhs
) const;
template Vector<double> MatrixDense<double>::frwd_sub(
    const Vector<double> &arg_rhs
) const;

}
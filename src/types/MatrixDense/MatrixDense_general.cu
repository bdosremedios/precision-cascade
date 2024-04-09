#include "types/MatrixDense/MatrixDense.h"

template <typename T>
Vector<T> MatrixDense<T>::frwd_sub(const Vector<T> &arg_rhs) const {

    dim3 block_dim(matrixdense_kernels::WARPSIZE, matrixdense_kernels::WARPSIZE);

    if (m_rows != n_cols) {
        throw std::runtime_error("MatrixDense::frwd_sub: non-square matrix");
    }
    if (m_rows != arg_rhs.rows()) {
        throw std::runtime_error("MatrixDense::frwd_sub: incompatible matrix and rhs");
    }

    Vector<T> soln(arg_rhs);

    T *d_soln = soln.d_vec;
    
    int n_blk = std::ceil(static_cast<float>(m_rows)/matrixdense_kernels::WARPSIZE);

    for (int i=0; i<n_blk; ++i) {

        matrixdense_kernels::lowtri_blk_solve_warp<T><<<1, matrixdense_kernels::WARPSIZE>>>(
            d_mat, m_rows, i*matrixdense_kernels::WARPSIZE, d_soln
        );

        dim3 grid_dim(1, n_blk-1-i);
        matrixdense_kernels::lowtri_rect_update_warp<T><<<grid_dim, block_dim>>>(
            d_mat, m_rows, i*matrixdense_kernels::WARPSIZE, d_soln
        );

    }

    return soln;

}

template Vector<__half> MatrixDense<__half>::frwd_sub(const Vector<__half> &arg_rhs) const;
template Vector<float> MatrixDense<float>::frwd_sub(const Vector<float> &arg_rhs) const;
template Vector<double> MatrixDense<double>::frwd_sub(const Vector<double> &arg_rhs) const;
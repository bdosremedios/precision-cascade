#include "types/MatrixDense.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

MatrixDense<double> MatrixDense<double>::operator*(const Scalar<double> &scalar) const {

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows*n_cols,
            scalar.d_scalar, CUDA_R_64F,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;
}

Vector<double> MatrixDense<double>::operator*(const Vector<double> &vec) const {

    if (vec.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid vec in matrix-vector prod (operator*(const Vector<double> &vec))"
        );
    }

    Vector<double> c(Vector<double>::Zero(handle, m_rows));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, 1, n_cols,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, n_cols,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

Vector<double> MatrixDense<double>::transpose_prod(const Vector<double> &vec) const {

    if (vec.rows() != m_rows) { throw std::runtime_error("MatrixDense: invalid vec in transpose_prod"); }

    Vector<double> c(Vector<double>::Zero(handle, n_cols));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n_cols, 1, m_rows,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            vec.d_vec, CUDA_R_64F, m_rows,
            SCALAR_ZERO_D.d_scalar,
            c.d_vec, CUDA_R_64F, n_cols,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<double> MatrixDense<double>::operator*(const MatrixDense<double> &mat) const {

    if (mat.rows() != n_cols) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix-matrix prod (operator*(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(MatrixDense<double>::Zero(handle, m_rows, mat.cols()));

    check_cublas_status(
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m_rows, mat.cols(), n_cols,
            SCALAR_ONE_D.d_scalar,
            d_mat, CUDA_R_64F, m_rows,
            mat.d_mat, CUDA_R_64F, n_cols,
            SCALAR_ZERO_D.d_scalar,
            c.d_mat, CUDA_R_64F, m_rows,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT
        )
    );

    return c;

}

MatrixDense<double> MatrixDense<double>::operator+(const MatrixDense<double> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix add (operator+(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            SCALAR_ONE_D.d_scalar, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

MatrixDense<double> MatrixDense<double>::operator-(const MatrixDense<double> &mat) const {

    if ((mat.rows() != m_rows) || (mat.cols() != n_cols)) {
        throw std::runtime_error(
            "MatrixDense: invalid mat in matrix subtract (operator-(const MatrixDense<double> &mat))"
        );
    }

    MatrixDense<double> c(*this);

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows*n_cols,
            SCALAR_MINUS_ONE_D.d_scalar, CUDA_R_64F,
            mat.d_mat, CUDA_R_64F, 1,
            c.d_mat, CUDA_R_64F, 1,
            CUDA_R_64F
        )
    );

    return c;

}

Scalar<double> MatrixDense<double>::norm() const {

    Scalar<double> result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows*n_cols,
            d_mat, CUDA_R_64F, 1,
            result.d_scalar, CUDA_R_64F,
            CUDA_R_64F
        )
    );

    return result;

}

namespace matdense_dbl_kern
{
    __global__ void solve_pivot_and_find_alpha(double *rhs, double *diag, double *alpha) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        rhs[tid] = rhs[tid]/diag[tid];
        alpha[tid] = -rhs[tid];
    }

    __global__ void cast_to_half(double *mat_src, half *mat_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            mat_dest[tid] = __double2half(mat_src[tid]);
        }
    }

    __global__ void cast_to_float(double *mat_src, float *mat_dest, int m) {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid < m) {
            mat_dest[tid] = __double2float_rn(mat_src[tid]);
        }
    }
}

Vector<double> MatrixDense<double>::back_sub(const Vector<double> &arg_rhs) const {

    if (m_rows != n_cols) {
        throw std::runtime_error("MatrixDense::back_sub: non-square matrix");
    }
    if (m_rows != arg_rhs.rows()) {
        throw std::runtime_error("MatrixDense::back_sub: incompatible matrix and rhs");
    }

    Vector<double> soln(arg_rhs);

    double *d_scale_val;
    check_cuda_error(cudaMalloc(&d_scale_val, sizeof(double)));

    for (int col=n_cols-1; col>=0; --col) {

        matdense_dbl_kern::solve_pivot_and_find_alpha<<<1, 1>>>(
            soln.d_vec+col, d_mat+(col+col*m_rows), d_scale_val
        );
        // std::cout << (col*m_rows) << std::endl;
        // double *h_a = static_cast<double *>(malloc(sizeof(double)));
        // double *h_b = static_cast<double *>(malloc(sizeof(double)));
        // double *h_c = static_cast<double *>(malloc(sizeof(double))); 
        // cudaMemcpy(h_a, d_scale_val, sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_b, soln.d_vec, sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_c, d_mat+(col*m_rows), sizeof(double), cudaMemcpyDeviceToHost);
        // std::cout << *h_a << " " << *h_b << " " << *h_c << " " << std::endl; 
        if (col > 0) {
            check_cublas_status(
                cublasAxpyEx(
                    handle, col,
                    d_scale_val, CUDA_R_64F,
                    d_mat+(col*m_rows), CUDA_R_64F, 1,
                    soln.d_vec, CUDA_R_64F, 1,
                    CUDA_R_64F
                )
            );
        }

    }

    check_cuda_error(cudaFree(d_scale_val));

    return soln;

}

Vector<double> MatrixDense<double>::frwd_sub(const Vector<double> &arg_rhs) const {

    if (m_rows != n_cols) {
        throw std::runtime_error("MatrixDense::frwd_sub: non-square matrix");
    }
    if (m_rows != arg_rhs.rows()) {
        throw std::runtime_error("MatrixDense::frwd_sub: incompatible matrix and rhs");
    }

    Vector<double> soln(arg_rhs);

    double *d_scale_val;
    check_cuda_error(cudaMalloc(&d_scale_val, sizeof(double)));

    for (int col=0; col<n_cols; ++col) {

        matdense_dbl_kern::solve_pivot_and_find_alpha<<<1, 1>>>(
            soln.d_vec+col, d_mat+(col*m_rows+col), d_scale_val
        );
        if (col < m_rows-1) {
            check_cublas_status(
                cublasAxpyEx(
                    handle, m_rows-1-col,
                    d_scale_val, CUDA_R_64F,
                    d_mat+(col*m_rows+(col+1)), CUDA_R_64F, 1,
                    soln.d_vec+(col+1), CUDA_R_64F, 1,
                    CUDA_R_64F
                )
            );
        }

    }

    check_cuda_error(cudaFree(d_scale_val));

    return soln;

}

MatrixDense<__half> MatrixDense<double>::to_half() const {

    MatrixDense<__half> created_mat(handle, m_rows, n_cols);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS))
    );
    matdense_dbl_kern::cast_to_half<<<NUM_THREADS, NUM_BLOCKS>>>(d_mat, created_mat.d_mat, m_rows*n_cols);

    return created_mat;

}

MatrixDense<float> MatrixDense<double>::to_float() const {

    MatrixDense<float> created_mat(handle, m_rows, n_cols);

    double NUM_THREADS = 1024; // threads per thread block just 1 warp
    double NUM_BLOCKS = static_cast<double>(
        std::ceil(static_cast<double>(m_rows*n_cols)/static_cast<double>(NUM_THREADS))
    );
    matdense_dbl_kern::cast_to_float<<<NUM_THREADS, NUM_BLOCKS>>>(d_mat, created_mat.d_mat, m_rows*n_cols);

    return created_mat;

}

MatrixDense<double> MatrixDense<double>::to_double() const { return MatrixDense<double>(*this); }
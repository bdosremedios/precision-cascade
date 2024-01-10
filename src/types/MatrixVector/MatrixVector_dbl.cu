#include "types/MatrixVector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
MatrixVector<double> MatrixVector<double>::operator*(const double &scalar) const {

    MatrixVector<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_64F, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
MatrixVector<double> & MatrixVector<double>::operator*=(const double &scalar) {

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_64F, d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return *this;

}

template<>
MatrixVector<double> MatrixVector<double>::operator+(const MatrixVector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    MatrixVector<double> c(*this);
    double alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
MatrixVector<double> MatrixVector<double>::operator-(const MatrixVector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    MatrixVector<double> c(*this);
    double alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
MatrixVector<double> & MatrixVector<double>::operator+=(const MatrixVector<double> &vec) {

    check_vecvec_op_compatibility(vec);

    double alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return *this;

}

template<>
MatrixVector<double> & MatrixVector<double>::operator-=(const MatrixVector<double> &vec) {

    check_vecvec_op_compatibility(vec);

    double alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return *this;

}

template<>
double MatrixVector<double>::dot(const MatrixVector<double> &vec) const {

    check_vecvec_op_compatibility(vec);
    
    double result;

    check_cublas_status(
        cublasDotEx(
            handle, m_rows, d_vec, CUDA_R_64F, 1, vec.d_vec, CUDA_R_64F, 1, &result, CUDA_R_64F, CUDA_R_64F
        )
    );

    return result;

}

template<>
double MatrixVector<double>::norm() const {

    double result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_64F, 1, &result, CUDA_R_64F, CUDA_R_64F
        )
    );

    return result;

}
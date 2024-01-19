#include "types/Vector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

template<>
Vector<double> Vector<double>::operator*(const double &scalar) const {

    Vector<double> c(*this);

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_64F, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
Vector<double> & Vector<double>::operator*=(const double &scalar) {

    check_cublas_status(
        cublasScalEx(
            handle, m_rows, &scalar, CUDA_R_64F, d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return *this;

}

template<>
Vector<double> Vector<double>::operator+(const Vector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<double> c(*this);
    double alpha = 1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
Vector<double> Vector<double>::operator-(const Vector<double> &vec) const {

    check_vecvec_op_compatibility(vec);

    Vector<double> c(*this);
    double alpha = -1.;

    check_cublas_status(
        cublasAxpyEx(
            handle, m_rows, &alpha, CUDA_R_64F, vec.d_vec, CUDA_R_64F, 1, c.d_vec, CUDA_R_64F, 1, CUDA_R_64F
        )
    );

    return c;

}

template<>
Vector<double> & Vector<double>::operator+=(const Vector<double> &vec) {

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
Vector<double> & Vector<double>::operator-=(const Vector<double> &vec) {

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
double Vector<double>::dot(const Vector<double> &vec) const {

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
double Vector<double>::norm() const {

    double result;

    check_cublas_status(
        cublasNrm2Ex(
            handle, m_rows, d_vec, CUDA_R_64F, 1, &result, CUDA_R_64F, CUDA_R_64F
        )
    );

    return result;

}
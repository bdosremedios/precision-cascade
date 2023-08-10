#ifndef SUBSTITUTION_H
#define SUBSTITUTION_H

#include "Eigen/Dense"

using Eigen::Matrix;

template <typename T>
Matrix<T, Dynamic, 1> back_substitution(
    Matrix<T, Dynamic, Dynamic> const &UT, Matrix<T, Dynamic, 1> const &rhs
) {

    // Check squareness and compatibility
    if (UT.rows() != UT.cols()) { throw runtime_error("Non square matrix in back substitution"); }
    if (UT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume UT is upper triangular
    Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::Zero(UT.cols(), 1);
    for (int i=UT.rows()-1; i>=0; --i) {
        if (UT(i, i) != 0) { // Skip if coefficient is zero
            x(i) = rhs(i);
            for (int j=i+1; j<=UT.rows()-1; ++j) {
                x(i) -= UT(i, j)*x(j);
            }
            x(i) /= UT(i, i);
        }
    }

    return x;

}

template <typename T>
Matrix<T, Dynamic, 1> frwd_substitution(
    Matrix<T, Dynamic, Dynamic> const &LT, Matrix<T, Dynamic, 1> const &rhs
) {

    // Check squareness and compatibility
    if (LT.rows() != LT.cols()) { throw runtime_error("Non square matrix in forward substitution"); }
    if (LT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume LT is lower triangular
    Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::Zero(LT.cols(), 1);
    for (int i=0; i<LT.rows(); ++i) {
        if (LT(i, i) != 0) { // Skip if coefficient is zero
            x(i) = rhs(i);
            for (int j=0; j<i; ++j) {
                x(i) -= LT(i, j)*x(j);
            }
            x(i) /= LT(i, i);
        }
    }

    return x;

}

#endif
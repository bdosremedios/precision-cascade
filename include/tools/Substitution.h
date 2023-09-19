#ifndef SUBSTITUTION_H
#define SUBSTITUTION_H

#include "Eigen/Dense"

#include <stdexcept>

using Eigen::Matrix;

using std::runtime_error;

template <template <typename> typename M, typename T>
MatrixVector<T> back_substitution(M<T> const &UT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (UT.rows() != UT.cols()) { throw runtime_error("Non square matrix in back substitution"); }
    if (UT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume UT is upper triangular
    MatrixVector<T> x = MatrixVector<T>::Zero(UT.cols());
    for (int i=UT.rows()-1; i>=0; --i) {
        if (UT.coeff(i, i) != 0) { // Skip if coefficient is zero
            x(i) = rhs(i);
            for (int j=i+1; j<=UT.rows()-1; ++j) {
                x(i) -= UT.coeff(i, j)*x(j);
            }
            x(i) /= UT.coeff(i, i);
        }
    }

    return x;

}

template <template <typename> typename M, typename T>
MatrixVector<T> frwd_substitution(M<T> const &LT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (LT.rows() != LT.cols()) { throw runtime_error("Non square matrix in forward substitution"); }
    if (LT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume LT is lower triangular
    MatrixVector<T> x = MatrixVector<T>::Zero(LT.cols());
    for (int i=0; i<LT.rows(); ++i) {
        if (LT.coeff(i, i) != 0) { // Skip if coefficient is zero
            x(i) = rhs(i);
            for (int j=0; j<i; ++j) {
                x(i) -= LT.coeff(i, j)*x(j);
            }
            x(i) /= LT.coeff(i, i);
        }
    }

    return x;

}

#endif
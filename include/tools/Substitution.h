#ifndef SUBSTITUTION_H
#define SUBSTITUTION_H

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "types/types.h" 

using std::runtime_error;

template <typename T>
MatrixVector<T> back_substitution(MatrixDense<T> const &UT, MatrixVector<T> const &rhs) {

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

template <typename T>
MatrixVector<T> back_substitution(MatrixSparse<T> const &UT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (UT.rows() != UT.cols()) { throw runtime_error("Non square matrix in back substitution"); }
    if (UT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume UT is upper triangular, iterate backwards through columns through non-zero entries
    // for backward substitution
    MatrixVector<T> x = rhs;
    for (int col=UT.cols()-1; col>=0; --col) {

        typename MatrixSparse<T>::ReverseInnerIterator it(UT, col);
        
        // Skip entries until reaching diagonal guarding against extra non-zeroes
        for (; it && (it.row() != it.col()); --it) { ; }
        if (it.row() != it.col()) { throw runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }

        x(col) /= it.value();
        --it;
        for (; it; --it) { x(it.row()) -= it.value()*x(col); }

    }

    return x;

}

template <typename T>
MatrixVector<T> frwd_substitution(MatrixDense<T> const &LT, MatrixVector<T> const &rhs) {

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

template <typename T>
MatrixVector<T> frwd_substitution(MatrixSparse<T> const &LT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (LT.rows() != LT.cols()) { throw runtime_error("Non square matrix in back substitution"); }
    if (LT.rows() != rhs.rows()) { throw runtime_error("Incompatible matrix and rhs"); }

    // Assume LT is lower triangular, iterate forwards through columns through non-zero entries
    // for forward substitution
    MatrixVector<T> x = rhs;
    for (int col=0; col<LT.cols(); ++col) {

        typename MatrixSparse<T>::InnerIterator it(LT, col);
        
        // Skip entries until reaching diagonal guarding against extra non-zeroes
        for (; it && (it.row() != it.col()); ++it) { ; }
        if (it.row() != it.col()) { throw runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }
        
        x(col) /= it.value();
        ++it;
        for (; it; ++it) { x(it.row()) -= it.value()*x(col); }

    }

    return x;

}

#endif
#ifndef SUBSTITUTION_H
#define SUBSTITUTION_H

#include "types/types.h"

template <typename T>
MatrixVector<T> back_substitution(MatrixDense<T> const &UT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (UT.rows() != UT.cols()) { throw std::runtime_error("Non square matrix in back substitution"); }
    if (UT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

    // Assume UT is upper triangular
    MatrixVector<T> x(rhs);
    for (int i=UT.rows()-1; i>=0; --i) {
        if (UT.get_elem(i, i) != 0) { // Skip if coefficient is zero
            for (int j=i+1; j<=UT.rows()-1; ++j) {
                x.set_elem(i, x.get_elem(i)-UT.get_elem(i, j)*x.get_elem(j));
            }
            x.set_elem(i, x.get_elem(i)/UT.get_elem(i, i));
        } else {
            x.set_elem(i, static_cast<T>(0));
        }
    }

    return x;

}

// template <typename T>
// MatrixVector<T> back_substitution(MatrixSparse<T> const &UT, MatrixVector<T> const &rhs) {

//     // Check squareness and compatibility
//     if (UT.rows() != UT.cols()) { throw std::runtime_error("Non square matrix in back substitution"); }
//     if (UT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

//     // Assume UT is upper triangular, iterate backwards through columns through non-zero entries
//     // for backward substitution
//     MatrixVector<T> x = rhs;
//     for (int col=UT.cols()-1; col>=0; --col) {

//         typename MatrixSparse<T>::ReverseInnerIterator it(UT, col);
        
//         // Skip entries until reaching diagonal guarding against extra non-zeroes
//         for (; it && (it.row() != it.col()); --it) { ; }
//         if (it.row() != it.col()) { throw std::runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }

//         x(col) /= it.value();
//         --it;
//         for (; it; --it) { x(it.row()) -= it.value()*x(col); }

//     }

//     return x;

// }

template <typename T>
MatrixVector<T> frwd_substitution(MatrixDense<T> const &LT, MatrixVector<T> const &rhs) {

    // Check squareness and compatibility
    if (LT.rows() != LT.cols()) { throw std::runtime_error("Non square matrix in forward substitution"); }
    if (LT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

    // Assume LT is lower triangular
    MatrixVector<T> x(rhs);
    for (int i=0; i<LT.rows(); ++i) {
        if (LT.get_elem(i, i) != 0) {
            for (int j=0; j<i; ++j) {
                x.set_elem(i, x.get_elem(i)-LT.get_elem(i, j)*x.get_elem(j));
            }
            x.set_elem(i, x.get_elem(i)/LT.get_elem(i, i));
        } else {
            x.set_elem(i, static_cast<T>(0));
        }
    }

    return x;

}

// template <typename T>
// MatrixVector<T> frwd_substitution(MatrixSparse<T> const &LT, MatrixVector<T> const &rhs) {

//     // Check squareness and compatibility
//     if (LT.rows() != LT.cols()) { throw std::runtime_error("Non square matrix in back substitution"); }
//     if (LT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

//     // Assume LT is lower triangular, iterate forwards through columns through non-zero entries
//     // for forward substitution
//     MatrixVector<T> x = rhs;
//     for (int col=0; col<LT.cols(); ++col) {

//         typename MatrixSparse<T>::InnerIterator it(LT, col);
        
//         // Skip entries until reaching diagonal guarding against extra non-zeroes
//         for (; it && (it.row() != it.col()); ++it) { ; }
//         if (it.row() != it.col()) { throw std::runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }
        
//         x(col) /= it.value();
//         ++it;
//         for (; it; ++it) { x(it.row()) -= it.value()*x(col); }

//     }

//     return x;

// }

#endif
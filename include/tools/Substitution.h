#ifndef SUBSTITUTION_H
#define SUBSTITUTION_H

#include "Eigen/Dense"

using Eigen::Matrix;

template <typename T>
Matrix<T, Dynamic, 1> back_substitution(
    const Matrix<T, Dynamic, Dynamic> &UT, const Matrix<T, Dynamic, 1> &rhs, const int solve_size
) {

    Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::Zero(solve_size, 1);
    for (int i=solve_size-1; i>=0; --i) {
        x(i) = rhs(i);
        for (int j=i+1; j<=solve_size-1; ++j) {
            x(i) -= UT(i, j)*x(j);
        }
        x(i) /= UT(i, i);
    }

    return x;

}

// template <typename T>
// Matrix<T, Dynamic, 1> fwrd_substitution(Matrix<T, Dynamic, Dynamic> &LT, Matrix<T, Dynamic, 1> rhs) {

// }

#endif
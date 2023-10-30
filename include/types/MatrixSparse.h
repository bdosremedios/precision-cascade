#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "Eigen/Dense"
#include "Eigen/SparseCore"

#include <cmath>

using Eigen::Matrix;
using Eigen::SparseMatrix;
using Eigen::Dynamic;

using std::min;

#include <iostream>
using std::cout, std::endl;

template <typename T>
class MatrixSparse: public SparseMatrix<T>
{
private:

    using Parent = SparseMatrix<T>;

public:

    // *** Constructor ***
    using Parent::SparseMatrix;

    // *** Element Access Methods ***
    const T coeff(int row, int col) const { return Parent::coeff(row, col); }
    T& coeffRef(int row, int col) { return Parent::coeffRef(row, col); }
    auto col(int _col) { return Parent::col(_col); } // auto to use whatever representation is used
                                                     // for block column in underlying matrix structure

    // *** Dimensions Methods ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }

    // *** Creation Methods ***
    static MatrixSparse<T> Random(int m, int n) {
        return Matrix<T, Dynamic, Dynamic>::Random(m, n).sparseView();
    }
    static MatrixSparse<T> Identity(int m, int n) {
        Parent temp = Parent(m, n);
        for (int i=0; i<min(m, n); ++i) { temp.coeffRef(i, i) = static_cast<T>(1); }
        return temp;
    }
    static MatrixSparse<T> Ones(int m, int n) {
        return Matrix<T, Dynamic, Dynamic>::Ones(m, n).sparseView();
    }
    static MatrixSparse<T> Zero(int m, int n) { return Parent(m, n); }

    // *** Resizing Methods ***
    void reduce() { Parent::prune(static_cast<T>(0)); }

    // *** Boolean Methods ***
    bool operator==(const MatrixSparse<T> &rhs) const {
        return Parent::isApprox(rhs);
    }

};

#endif
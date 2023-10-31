#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include <Eigen/Dense>

#include "MatrixVector.h"

using Eigen::Matrix;
using Eigen::Dynamic;

#include <iostream>
using std::cout, std::endl;

template <typename T>
class MatrixDense: public Matrix<T, Dynamic, Dynamic>
{
private:

    using Parent = Matrix<T, Dynamic, Dynamic>;

public:

    // *** Constructor ***
    using Parent::Matrix;

    // *** Element Access Methods ***
    const T coeff(int row, int col) const { return Parent::operator()(row, col); }
    T& coeffRef(int row, int col) { return Parent::operator()(row, col); }

    // auto to use arbitrary block representation (reqs block assignment & assignment/conversion to MatrixDense)
    auto col(int _col) { return Parent::col(_col); } 
    auto block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Dimensions Methods ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }

    // *** Creation Methods ***
    static MatrixDense<T> Random(int m, int n) { return Parent::Random(m, n); }
    static MatrixDense<T> Identity(int m, int n) { return Parent::Identity(m, n); }
    static MatrixDense<T> Ones(int m, int n) { return Parent::Ones(m, n); }
    static MatrixDense<T> Zero(int m, int n) { return Parent::Zero(m, n); }

    // *** Resizing Methods ***

    // Resize without altering existing entries
    void conservativeResize(int m, int n) { Parent::conservativeResize(m, n); }
    void reduce() { ; } // Do nothing on reduction

    // *** Boolean Methods ***
    bool operator==(const MatrixDense<T> &rhs) const { return Parent::operator==(rhs); }

    // *** Cast Methods ***
    template <typename Cast_T>
    MatrixDense<Cast_T> cast() const { return Parent::template cast<Cast_T>(); }

    // *** Calculation/Assignment Methods ***
    MatrixDense<T> transpose() { return Parent::transpose(); }
    MatrixDense<T> operator*(const T &scalar) const { return Parent::operator*(scalar); }
    MatrixDense<T> operator/(const T &scalar) const { return Parent::operator/(scalar); }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const { return Parent::operator*(vec); }
    MatrixDense<T> operator*(const MatrixDense<T> &mat) const { return Parent::operator*(mat); } // Needed for testing

};

#endif
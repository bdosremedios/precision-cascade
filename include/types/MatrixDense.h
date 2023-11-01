#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include <Eigen/Dense>

#include "MatrixVector.h"

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T>
class MatrixDense: public Matrix<T, Dynamic, Dynamic>
{
private:

    using Parent_MD = Matrix<T, Dynamic, Dynamic>;

public:

    // *** Constructor ***
    using Parent_MD::Matrix;

    // *** Element Access Methods ***
    const T coeff(int row, int col) const { return Parent_MD::operator()(row, col); }
    T& coeffRef(int row, int col) { return Parent_MD::operator()(row, col); }

    // auto to use arbitrary block representation (reqs block assignment & assignment/conversion to MatrixDense)
    auto col(int _col) { return Parent_MD::col(_col); } 
    auto block(int row, int col, int m, int n) { return Parent_MD::block(row, col, m, n); }

    // *** Dimensions Methods ***
    int rows() const { return Parent_MD::rows(); }
    int cols() const { return Parent_MD::cols(); }

    // *** Creation Methods ***
    static MatrixDense<T> Random(int m, int n) { return Parent_MD::Random(m, n); }
    static MatrixDense<T> Identity(int m, int n) { return Parent_MD::Identity(m, n); }
    static MatrixDense<T> Ones(int m, int n) { return Parent_MD::Ones(m, n); }
    static MatrixDense<T> Zero(int m, int n) { return Parent_MD::Zero(m, n); }

    // *** Resizing Methods ***

    // Resize without altering existing entries
    void conservativeResize(int m, int n) { Parent_MD::conservativeResize(m, n); }
    void reduce() { ; } // Do nothing on reduction

    // *** Boolean Methods ***
    bool operator==(const MatrixDense<T> &rhs) const { return Parent_MD::operator==(rhs); }

    // *** Cast Methods ***
    template <typename Cast_T>
    MatrixDense<Cast_T> cast() const { return Parent_MD::template cast<Cast_T>(); }

    // *** Calculation/Assignment Methods ***
    MatrixDense<T> transpose() { return Parent_MD::transpose(); }
    MatrixDense<T> operator*(const T &scalar) const { return Parent_MD::operator*(scalar); }
    MatrixDense<T> operator/(const T &scalar) const { return Parent_MD::operator/(scalar); }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const { return Parent_MD::operator*(vec.base()); }
    MatrixDense<T> operator*(const MatrixDense<T> &mat) const { return Parent_MD::operator*(mat); } // Needed for testing

};

#endif
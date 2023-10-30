#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include "Eigen/Dense"

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
    auto col(int _col) { return Parent::col(_col); } // auto to use whatever representation is used
                                                     // for block column in underlying matrix structure

    // *** Dimensions Methods ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }

    // *** Creation Methods ***
    static MatrixDense<T> Random(int m, int n) { return Parent::Random(m, n); }
    static MatrixDense<T> Identity(int m, int n) { return Parent::Identity(m, n); }
    static MatrixDense<T> Ones(int m, int n) { return Parent::Ones(m, n); }
    static MatrixDense<T> Zero(int m, int n) { return Parent::Zero(m, n); }

    // *** Resizing Methods ***
    void reduce() { ; } // Do nothing on reduction

};

#endif
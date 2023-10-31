#ifndef VECTOR_H
#define VECTOR_H

#include "Eigen/Dense"

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T>
class MatrixVector: public Matrix<T, Dynamic, 1>
{
private:

    using Parent = Matrix<T, Dynamic, 1>;

    static void check_n(int n) { if (n != 1) { throw std::runtime_error("Invalid number of columns for vector."); } }

public:

    // *** Constructor ***
    using Parent::Matrix;
    MatrixVector(int m, int n): Parent::Matrix(m, 1) { check_n(n); }
    MatrixVector(int m): Parent::Matrix(m, 1) {}

    // *** Element Access Methods ***
    // auto to use arbitrary block representation (reqs block assignment & assignment/conversion to MatrixDense)
    auto block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Dimensions Methods ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }

    // *** Creation Methods ***
    static MatrixVector<T> Zero(int m, int n) { check_n(n); return Parent::Zero(m, 1); }
    static MatrixVector<T> Zero(int m) { return Parent::Zero(m, 1); }
    static MatrixVector<T> Ones(int m, int n) { check_n(n); return Parent::Ones(m, 1); }
    static MatrixVector<T> Ones(int m) { return Parent::Ones(m, 1); }
    static MatrixVector<T> Random(int m, int n) { check_n(n); return Parent::Random(m, 1); }
    static MatrixVector<T> Random(int m) { return Parent::Random(m, 1); }

    // *** Resizing Methods ***
    void reduce() { ; }

    // *** Cast Methods ***
    template <typename Cast_T>
    MatrixVector<Cast_T> cast() const { return Parent::template cast<Cast_T>(); }

    // *** Calculation Methods ***
    T dot(const MatrixVector<T> &vec) const { return Parent::dot(vec); }
    T norm() const { return Parent::norm(); }
    MatrixVector<T> operator*(const T &scalar) const { return Parent::operator*(scalar); }
    MatrixVector<T> operator/(const T &scalar) const { return Parent::operator/(scalar); }
    MatrixVector<T> operator-(const MatrixVector<T> &vec) const { return Parent::operator-(vec); }
    MatrixVector<T> operator+(const MatrixVector<T> &vec) const { return Parent::operator+(vec); }

};

#endif
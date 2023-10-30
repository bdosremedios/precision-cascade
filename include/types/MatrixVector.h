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

    using Parent::Matrix;
    MatrixVector(int m, int n): Parent::Matrix(m, 1) { check_n(n); }
    MatrixVector(int m): Parent::Matrix(m, 1) {}

    static MatrixVector<T> Zero(int m, int n) { check_n(n); return Parent::Zero(m, 1); }
    static MatrixVector<T> Zero(int m) { return Parent::Zero(m, 1); }
    static MatrixVector<T> Ones(int m, int n) { check_n(n); return Parent::Ones(m, 1); }
    static MatrixVector<T> Ones(int m) { return Parent::Ones(m, 1); }
    static MatrixVector<T> Random(int m, int n) { check_n(n); return Parent::Random(m, 1); }
    static MatrixVector<T> Random(int m) { return Parent::Random(m, 1); }

    // *** Resizing Methods ***
    void reduce() { ; }

};

#endif
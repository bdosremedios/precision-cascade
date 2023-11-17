#ifndef MATRIXVECTOR_H
#define MATRIXVECTOR_H

#include <Eigen/Dense>
#include <iostream>

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T> class MatrixDense;
template <typename T> class MatrixSparse;

template <typename T>
class MatrixVector: private Matrix<T, Dynamic, 1>
{
private:

    using Parent = Matrix<T, Dynamic, 1>;

    static void check_n(int n) { if (n != 1) { throw std::runtime_error("Invalid number of columns for vector."); } }

    // Only allow MatrixDense and MatrixSparse to access private methods like base
    friend MatrixDense<T>;
    friend MatrixSparse<T>;
    const Parent &base() const { return *this; }

public:

    // *** Construction/Assignment/Destruction ***
    MatrixVector(): Parent::Matrix(0, 1) {}
    MatrixVector(int m, int n): Parent::Matrix(m, 1) { check_n(n); }
    MatrixVector(int m): Parent::Matrix(m, 1) {}
    MatrixVector(const MatrixVector<T> &vec) = default;
    MatrixVector(const Parent &parent): Parent::Matrix(parent) { check_n(parent.cols()); }
    MatrixVector(const typename MatrixDense<T>::Col &col): Parent::Matrix(col.base()) {}
    MatrixVector(const typename MatrixSparse<T>::Col &col): Parent::Matrix(col.base()) {}
    MatrixVector& operator=(const MatrixVector &vec) = default;
    virtual ~MatrixVector() = default;

    // *** Element Access ***
    const T coeff(int row, int col) const {
        if (col > 0) { throw std::runtime_error("Invalid column access for vector."); }
        return Parent::operator()(row, col);
    }
    T& coeffRef(int row, int col) {
        if (col > 0) { throw std::runtime_error("Invalid column access for vector."); }
        return Parent::operator()(row, col);
    }
    const T operator()(int row) const { return coeff(row, 0); }
    T& operator()(int row) { return coeffRef(row, 0); }
    MatrixVector<T> slice(int start, int elements) const { 
        return typename Parent::Matrix(Parent::block(start, 0, elements, 1));
    }

    // *** Static Creation ***
    static MatrixVector<T> Zero(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Zero(m, 1));
    }
    static MatrixVector<T> Zero(int m) { return Zero(m, 1); }
    static MatrixVector<T> Ones(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Ones(m, 1));
    }
    static MatrixVector<T> Ones(int m) { return Ones(m, 1); }
    static MatrixVector<T> Random(int m, int n) {
        check_n(n);
        return typename Parent::Matrix(Parent::Random(m, 1));
    }
    static MatrixVector<T> Random(int m) { return Random(m, 1); }

    // *** Properties ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }
    void print() const { std::cout << *this << std::endl << std::endl; }

    // *** Resizing ***
    void reduce() { ; }

    // *** Boolean ***
    bool operator==(const MatrixVector<T> &rhs) const { return Parent::isApprox(rhs); }

    // *** Explicit Cast ***
    template <typename Cast_T>
    MatrixVector<Cast_T> cast() const {
        return typename Matrix<Cast_T, Dynamic, 1>::Matrix(Parent::template cast<Cast_T>());
    }

    // *** Arithmetic and Compound Operations ***
    T dot(const MatrixVector<T> &vec) const { return Parent::dot(vec); }
    T norm() const { return Parent::norm(); }
    MatrixVector<T> operator*(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator*(scalar));
    }
    MatrixVector<T> operator/(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator/(scalar));
    }
    MatrixVector<T> operator-(const MatrixVector<T> &vec) const {
        return typename Parent::Matrix(Parent::operator-(vec));
    }
    MatrixVector<T> operator+(const MatrixVector<T> &vec) const {
        return typename Parent::Matrix(Parent::operator+(vec));
    }
    MatrixVector<T> operator+=(const MatrixVector<T> &vec) { return *this = *this + vec; };
    MatrixVector<T> operator-=(const MatrixVector<T> &vec) { return *this = *this - vec; };
};

#include "MatrixDense.h"
#include "MatrixSparse.h"

#endif
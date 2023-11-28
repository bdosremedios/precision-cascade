#ifndef MATRIXSPARSE_H
#define MATRIXSPARSE_H

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "MatrixVector.h"
#include "MatrixDense.h"

#include <cmath>
#include <iostream>

using Eigen::Matrix;
using Eigen::SparseMatrix;
using Eigen::Dynamic;

using std::min;

template <typename T>
class MatrixSparse: private SparseMatrix<T>
{
private:

    using Parent = SparseMatrix<T>;

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Constructors ***
    MatrixSparse(): Parent::SparseMatrix(0, 0) {}
    MatrixSparse(int m, int n): Parent::SparseMatrix(m, n) {}
    MatrixSparse(const Parent &parent): Parent::SparseMatrix(parent) {}

    // *** Element Access ***
    const T coeff(int row, int col) const { return Parent::coeff(row, col); }
    T& coeffRef(int row, int col) { return Parent::coeffRef(row, col); }
    Col col(int _col) { return Parent::col(_col); }
    Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Properties ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }
    void print() const { std::cout << *this << std::endl << std::endl; }

    // *** Static Creation ***
    static MatrixSparse<T> Random(int m, int n) {
        return typename Parent::SparseMatrix(Matrix<T, Dynamic, Dynamic>::Random(m, n).sparseView());
    }
    static MatrixSparse<T> Identity(int m, int n) {
        Parent temp = Parent(m, n);
        for (int i=0; i<min(m, n); ++i) { temp.coeffRef(i, i) = static_cast<T>(1); }
        return temp;
    }
    static MatrixSparse<T> Ones(int m, int n) {
        return typename Parent::SparseMatrix(Matrix<T, Dynamic, Dynamic>::Ones(m, n).sparseView());
    }
    static MatrixSparse<T> Zero(int m, int n) { return Parent(m, n); }

    // *** Resizing ***
    void reduce() { Parent::prune(static_cast<T>(0)); }

    // *** Boolean ***
    bool operator==(const MatrixSparse<T> &rhs) const { return Parent::isApprox(rhs); }

    // *** Explicit Cast ***
    template <typename Cast_T>
    MatrixSparse<Cast_T> cast() const {
        return typename Eigen::SparseMatrix<Cast_T>::SparseMatrix(
            Parent::template cast<Cast_T>()
        );
    }

    // *** Arithmetic and Compound Operations ***
    MatrixSparse<T> transpose() const {
        return typename Parent::SparseMatrix(Parent::transpose());
    }
    MatrixSparse<T> operator*(const T &scalar) const {
        return typename Parent::SparseMatrix(Parent::operator*(scalar));
    }
    MatrixSparse<T> operator/(const T &scalar) const {
        return typename Parent::SparseMatrix(Parent::operator/(scalar));
    }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const {
        return typename Matrix<T, Dynamic, 1>::Matrix(Parent::operator*(vec.base()));
    }
    T norm() const { return Parent::norm(); } // Needed for testing
    MatrixSparse<T> operator-(const MatrixSparse<T> &mat) const { // Needed for testing
        return typename Parent::SparseMatrix(Parent::operator-(mat));
    }
    MatrixSparse<T> operator+(const MatrixSparse<T> &mat) const { // Needed for testing
        return typename Parent::SparseMatrix(Parent::operator+(mat));
    }
    MatrixSparse<T> operator*(const MatrixSparse<T> &mat) const { // Needed for testing
        return typename Parent::SparseMatrix(Parent::operator*(mat));
    }

    // Forward iterator over sparse inner columns skipping zeroes
    class InnerIterator: public Parent::InnerIterator {

        public:
            
            InnerIterator(const MatrixSparse<T> &mat, int start):
                Parent::InnerIterator(mat, start)
            {}

            int col() { return Parent::InnerIterator::col(); }
            int row() { return Parent::InnerIterator::row(); }
            T value() { return Parent::InnerIterator::value(); }
            typename Parent::InnerIterator &operator++() {
                return Parent::InnerIterator::operator++();
            }
            operator bool() const { return Parent::InnerIterator::operator bool(); }

    };

    // Reverse iterator over sparse inner columns skipping zeroes
    class ReverseInnerIterator: public Parent::ReverseInnerIterator {

        public:
            
            ReverseInnerIterator(const MatrixSparse<T> &mat, int start):
                Parent::ReverseInnerIterator(mat, start)
            {}

            int col() { return Parent::ReverseInnerIterator::col(); }
            int row() { return Parent::ReverseInnerIterator::row(); }
            T value() { return Parent::ReverseInnerIterator::value(); }
            typename Parent::ReverseInnerIterator &operator--() {
                return Parent::ReverseInnerIterator::operator--();
            }
            operator bool() const { return Parent::ReverseInnerIterator::operator bool(); }

    };

    // Nested class representing sparse matrix column
    // NEEDS CREATION-FROM/CAST-TO MatrixVector<T>
    class Col: private Eigen::Block<Parent, Eigen::Dynamic, 1, true> {

        private:
            using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;
            friend MatrixVector<T>;
            const ColParent &base() const { return *this; }

        public:
            Col(const ColParent &other): ColParent(other) {}
            Col operator=(const MatrixVector<T> vec) { return ColParent::operator=(vec.base().sparseView()); }
            T norm() const { return ColParent::norm(); }

    };

    // Nested class representing sparse matrix block
    // NEEDS CREATION FROM/CAST TO MatrixDense<T>
    class Block: private Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic> {

        private:
            using BlockParent = Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>;
            friend MatrixDense<T>;
            const BlockParent &base() const { return *this; }

        public:
            Block(const BlockParent &other): BlockParent(other) {}
            Block operator=(const MatrixVector<T> vec) { return BlockParent::operator=(vec.base()); }
            Block operator=(const MatrixDense<T> &mat) { return BlockParent::operator=(mat); }

    };

    
};

#endif
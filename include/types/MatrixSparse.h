#ifndef MATRIXSPARSE_H
#define MATRIXSPARSE_H

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "MatrixVector.h"

#include <cmath>

using Eigen::Matrix;
using Eigen::SparseMatrix;
using Eigen::Dynamic;

using std::min;

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

    // auto to use arbitrary block representation (reqs block assignment & assignment/conversion to MatrixDense)
    class Block; class Col;
    Col col(int _col) { return Parent::col(_col); }
    Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

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
    bool operator==(const MatrixSparse<T> &rhs) const { return Parent::isApprox(rhs); }

    // *** Cast Methods ***
    template <typename Cast_T>
    MatrixSparse<Cast_T> cast() const { return Parent::template cast<Cast_T>(); }

    // *** Calculation/Assignment Methods ***
    MatrixSparse<T> transpose() { return Parent::transpose(); }
    MatrixSparse<T> operator*(const T &scalar) const { return Parent::operator*(scalar); }
    MatrixSparse<T> operator/(const T &scalar) const { return Parent::operator/(scalar); }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const {
        return typename Matrix<T, Dynamic, 1>::Matrix(Parent::operator*(vec.base()));
    }
    MatrixSparse<T> operator*(const MatrixSparse<T> &mat) const { return Parent::operator*(mat); } // Needed for testing

    // Forward iterator over sparse inner columns, to iterate efficienctly over non-zeros
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

    // Reverse iterator over sparse inner columns, to iterate efficienctly over non-zeros
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

    class Col: public Eigen::Block<Parent, Eigen::Dynamic, 1, true> {

        private:
            using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;

        public:
            Col(const ColParent &other): ColParent(other) {}
            Col operator=(const MatrixVector<T> vec) { return ColParent::operator=(vec.base().sparseView()); }

    };

    class Block: public Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic> {

        private:
            using BlockParent = Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>;

        public:
            Block(const BlockParent &other): BlockParent(other) {}
            Block operator=(const MatrixVector<T> vec) { return BlockParent::operator=(vec.base()); }
            Block operator=(const MatrixDense<T> &mat) { return BlockParent::operator=(mat); }

    };

    
};

#endif
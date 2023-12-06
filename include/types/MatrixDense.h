#ifndef MATRIXDENSE_H
#define MATRIXDENSE_H

#include <Eigen/Dense>

#include "MatrixVector.h"

#include <iostream>

template <typename T> class MatrixSparse;

template <typename T>
class MatrixDense: private Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
{
private:

    using Parent = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Constructors ***
    MatrixDense(): Parent::Matrix(0, 0) {}
    MatrixDense(int m, int n): Parent::Matrix(m, n) {}

    MatrixDense(std::initializer_list<std::initializer_list<T>> li):
        MatrixDense(
            li.size(),
            (li.size() == 0) ? 0 : std::cbegin(li)->size()
        )
    {
        int i=0;
        for (auto curr_row = std::cbegin(li); curr_row != std::cend(li); ++curr_row) {
            int j=0;
            for (auto curr_elem = std::cbegin(*curr_row); curr_elem != std::cend(*curr_row); ++curr_elem) {
                if (j >= cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
                this->coeffRef(i, j) = *curr_elem;
                ++j;
            }
            if (j != cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
            ++i;
        }
    }

    MatrixDense(const Parent &parent): Parent::Matrix(parent) {}
    MatrixDense(const Block &block): Parent::Matrix(block.base()) {}
    MatrixDense(const typename MatrixSparse<T>::Block &block): Parent::Matrix(block.base()) {}

    // *** Cast ***
    SparseMatrix<T> sparse() const { return SparseMatrix<T>(Parent::sparseView()); };

    // *** Element Access ***
    const T coeff(int row, int col) const { return Parent::operator()(row, col); }
    T& coeffRef(int row, int col) { return Parent::operator()(row, col); }
    Col col(int _col) { return Parent::col(_col); } 
    Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Properties ***
    int rows() const { return Parent::rows(); }
    int cols() const { return Parent::cols(); }
    void print() const { std::cout << *this << std::endl << std::endl; }

    // *** Static Creation ***
    static MatrixDense<T> Random(int m, int n) { return typename Parent::Matrix(Parent::Random(m, n)); }
    static MatrixDense<T> Identity(int m, int n) { return typename Parent::Matrix(Parent::Identity(m, n)); }
    static MatrixDense<T> Ones(int m, int n) { return typename Parent::Matrix(Parent::Ones(m, n)); }
    static MatrixDense<T> Zero(int m, int n) { return typename Parent::Matrix(Parent::Zero(m, n)); }

    // *** Resizing ***
    void reduce() { ; } // Do nothing on reduction

    // *** Boolean ***
    bool operator==(const MatrixDense<T> &rhs) const { return Parent::operator==(rhs); }

    // *** Explicit Cast ***
    template <typename Cast_T>
    MatrixDense<Cast_T> cast() const {
        return typename Eigen::Matrix<Cast_T, Eigen::Dynamic, Eigen::Dynamic>::Matrix(
            Parent::template cast<Cast_T>()
        );
    }

    // *** Arithmetic and Compound Operations ***
    MatrixDense<T> transpose() const {
        return typename Parent::Matrix(Parent::transpose());
    }
    MatrixDense<T> operator*(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator*(scalar));
    }
    MatrixDense<T> operator/(const T &scalar) const {
        return typename Parent::Matrix(Parent::operator/(scalar));
    }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const {
        return typename Eigen::Matrix<T, Eigen::Dynamic, 1>::Matrix(Parent::operator*(vec.base()));
    }
    T norm() const { return Parent::norm(); } // Needed for testing
    MatrixDense<T> operator-(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator-(mat));
    }
    MatrixDense<T> operator+(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator+(mat));
    }
    MatrixDense<T> operator*(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator*(mat));
    }

    // Nested class representing sparse matrix column
    // NEEDS CREATION-FROM/CAST-TO MatrixVector<T>
    class Col: private Eigen::Block<Parent, Eigen::Dynamic, 1, true> {

        private:
            using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;
            friend MatrixVector<T>;
            const ColParent &base() const { return *this; }

        public:
            Col(const ColParent &other): ColParent(other) {}
            Col(const Eigen::Block<const Parent, Eigen::Dynamic, 1, true> &other): ColParent(other) {}
            Col operator=(const MatrixVector<T> vec) { return ColParent::operator=(vec.base()); }
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
            Block(const MatrixDense<T> &mat): BlockParent(mat) {}
            Block operator=(const MatrixVector<T> vec) { return BlockParent::operator=(vec.base()); }
            Block operator=(const MatrixDense<T> &mat) { return BlockParent::operator=(mat); }

    };

};

#include "MatrixSparse.h"

#endif
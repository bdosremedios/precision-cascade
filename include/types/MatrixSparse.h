#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

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

    using Parent_MS = SparseMatrix<T>;

public:

    // *** Constructor ***
    using Parent_MS::SparseMatrix;

    // *** Element Access Methods ***

    const T coeff(int row, int col) const { return Parent_MS::coeff(row, col); }
    T& coeffRef(int row, int col) { return Parent_MS::coeffRef(row, col); }

    // auto to use arbitrary block representation (reqs block assignment & assignment/conversion to MatrixDense)
    auto col(int _col) { return Parent_MS::col(_col); }
    auto block(int row, int col, int m, int n) { return Parent_MS::block(row, col, m, n); }

    // *** Dimensions Methods ***
    int rows() const { return Parent_MS::rows(); }
    int cols() const { return Parent_MS::cols(); }

    // *** Creation Methods ***
    static MatrixSparse<T> Random(int m, int n) {
        return Matrix<T, Dynamic, Dynamic>::Random(m, n).sparseView();
    }
    static MatrixSparse<T> Identity(int m, int n) {
        Parent_MS temp = Parent_MS(m, n);
        for (int i=0; i<min(m, n); ++i) { temp.coeffRef(i, i) = static_cast<T>(1); }
        return temp;
    }
    static MatrixSparse<T> Ones(int m, int n) {
        return Matrix<T, Dynamic, Dynamic>::Ones(m, n).sparseView();
    }
    static MatrixSparse<T> Zero(int m, int n) { return Parent_MS(m, n); }

    // *** Resizing Methods ***
    void reduce() { Parent_MS::prune(static_cast<T>(0)); }

    // *** Boolean Methods ***
    bool operator==(const MatrixSparse<T> &rhs) const { return Parent_MS::isApprox(rhs); }

    // *** Cast Methods ***
    template <typename Cast_T>
    MatrixSparse<Cast_T> cast() const { return Parent_MS::template cast<Cast_T>(); }

    // *** Calculation/Assignment Methods ***
    MatrixSparse<T> transpose() { return Parent_MS::transpose(); }
    MatrixSparse<T> operator*(const T &scalar) const { return Parent_MS::operator*(scalar); }
    MatrixSparse<T> operator/(const T &scalar) const { return Parent_MS::operator/(scalar); }
    MatrixVector<T> operator*(const MatrixVector<T> &vec) const { return Parent_MS::operator*(vec.base()); }
    MatrixSparse<T> operator*(const MatrixSparse<T> &mat) const { return Parent_MS::operator*(mat); } // Needed for testing

    // Forward iterator over sparse inner columns, to iterate efficienctly over non-zeros
    class InnerIterator: public Parent_MS::InnerIterator {

        public:
            
            InnerIterator(const MatrixSparse<T> &mat, int start):
                Parent_MS::InnerIterator(mat, start)
            {}

            int col() { return Parent_MS::InnerIterator::col(); }
            int row() { return Parent_MS::InnerIterator::row(); }
            T value() { return Parent_MS::InnerIterator::value(); }
            typename Parent_MS::InnerIterator &operator++() {
                return Parent_MS::InnerIterator::operator++();
            }
            operator bool() const { return Parent_MS::InnerIterator::operator bool(); }

    };

    // Reverse iterator over sparse inner columns, to iterate efficienctly over non-zeros
    class ReverseInnerIterator: public Parent_MS::ReverseInnerIterator {

        public:
            
            ReverseInnerIterator(const MatrixSparse<T> &mat, int start):
                Parent_MS::ReverseInnerIterator(mat, start)
            {}

            int col() { return Parent_MS::ReverseInnerIterator::col(); }
            int row() { return Parent_MS::ReverseInnerIterator::row(); }
            T value() { return Parent_MS::ReverseInnerIterator::value(); }
            typename Parent_MS::ReverseInnerIterator &operator--() {
                return Parent_MS::ReverseInnerIterator::operator--();
            }
            operator bool() const { return Parent_MS::ReverseInnerIterator::operator bool(); }

    };
    
};

#endif
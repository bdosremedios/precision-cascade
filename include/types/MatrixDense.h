#ifndef MATRIXDENSE_H
#define MATRIXDENSE_H

#include <Eigen/Dense>

#include "MatrixVector.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

template <typename T> class MatrixSparse;

template <typename T>
class MatrixDense: private Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
{
private:

    using Parent = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    friend MatrixSparse<T>;
    const Parent &base() const { return *this; }

    int m, n;
    size_t mem_size;
    T *d_mat = nullptr;
    cublasHandle_t handle;

    void allocate_d_mat() {
        cudaMalloc(&d_mat, mem_size);
    }

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Basic Constructors ***
    MatrixDense(const cublasHandle_t &arg_handle):
        m(0), n(0), mem_size(m*n*sizeof(T))
    { allocate_d_mat(); }

    MatrixDense(const cublasHandle_t &arg_handle, int arg_m, int arg_n):
        m(arg_m), n(arg_n), mem_size(m*n*sizeof(T))
    { allocate_d_mat(); }

    // MatrixDense(std::initializer_list<std::initializer_list<T>> li):
    //     MatrixDense(li.size(), (li.size() == 0) ? 0 : std::cbegin(li)->size())
    // {
    //     int i=0;
    //     for (auto curr_row = std::cbegin(li); curr_row != std::cend(li); ++curr_row) {
    //         int j=0;
    //         for (auto curr_elem = std::cbegin(*curr_row); curr_elem != std::cend(*curr_row); ++curr_elem) {
    //             if (j >= cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
    //             this->coeffRef(i, j) = *curr_elem;
    //             ++j;
    //         }
    //         if (j != cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
    //         ++i;
    //     }
    //     if (i != rows()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
    // }

    // MatrixDense(const Parent &parent): Parent::Matrix(parent) {}
    // MatrixDense(const Block &block): Parent::Matrix(block.base()) {}
    // MatrixDense(const typename MatrixSparse<T>::Block &block): Parent::Matrix(block.base()) {}

    // *** Destructor ***
    ~MatrixDense() {
        cudaFree(d_mat);
        d_mat = nullptr;
    }

    // *** Cast ***
    MatrixSparse<T> sparse() const { return MatrixSparse<T>(Parent::sparseView()); };

    // *** Element Access ***
    const T coeff(int row, int col) const { return Parent::operator()(row, col); }
    T& coeffRef(int row, int col) { return Parent::operator()(row, col); }
    Col col(int _col) { return Parent::col(_col); } 
    Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Properties ***
    int rows() const { return m; }
    int cols() const { return n; }
    void print() { std::cout << *this << std::endl << std::endl; }

    // *** Static Creation ***
    static MatrixDense<T> Random(int m, int n) { return typename Parent::Matrix(Parent::Random(m, n)); }
    static MatrixDense<T> Identity(int m, int n) { return typename Parent::Matrix(Parent::Identity(m, n)); }
    static MatrixDense<T> Ones(int m, int n) { return typename Parent::Matrix(Parent::Ones(m, n)); }
    static MatrixDense<T> Zero(int m, int n) { return typename Parent::Matrix(Parent::Zero(m, n)); }

    // *** Resizing ***
    void reduce() { ; } // Do nothing on reduction

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
    MatrixDense<T> operator+(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator+(mat));
    }
    MatrixDense<T> operator-(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator-(mat));
    }
    MatrixDense<T> operator*(const MatrixDense<T> &mat) const { // Needed for testing
        return typename Parent::Matrix(Parent::operator*(mat));
    }

    // Nested class representing sparse matrix column
    // Requires: assignment from/cast to MatrixVector<T>
    class Col: private Eigen::Block<Parent, Eigen::Dynamic, 1, true>
    {
    private:

        using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;
        using ConstColParent = Eigen::Block<const Parent, Eigen::Dynamic, 1, true>;
        friend MatrixVector<T>;
        friend MatrixDense<T>;
        const ColParent &base() const { return *this; }
        Col(const ColParent &other): ColParent(other) {}
        Col(const ConstColParent &other): ColParent(other) {}

    public:

        Col operator=(const MatrixVector<T> vec) { return ColParent::operator=(vec.base()); }

    };

    // Nested class representing sparse matrix block
    // Requires: assignment from/cast to MatrixDense<T> and assignment from MatrixVector
    class Block: private Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>
    {
    private:

        using BlockParent = Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>;
        friend MatrixDense<T>;
        const BlockParent &base() const { return *this; }
        Block(const BlockParent &other): BlockParent(other) {}

    public:

        Block operator=(const MatrixVector<T> vec) { return BlockParent::operator=(vec.base()); }
        Block operator=(const MatrixDense<T> &mat) { return BlockParent::operator=(mat.base()); }

    };

};

#include "MatrixSparse.h"

#endif
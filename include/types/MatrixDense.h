#ifndef MATRIXDENSE_H
#define MATRIXDENSE_H

#include <Eigen/Dense>

#include "MatrixVector.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"

template <typename T> class MatrixSparse;

template <typename T>
class MatrixDense: private Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
{
private:

    using Parent = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    friend MatrixSparse<T>;
    // const Parent &base() const { return *this; }

    int m, n;
    size_t mem_size;
    T *d_mat = nullptr;
    cublasHandle_t handle;

    void allocate_d_mat() {
        check_cuda_error(cudaMalloc(&d_mat, mem_size));
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

    // Row-major nested initializer list assumed for intuitive user instantiation
    MatrixDense(const cublasHandle_t &arg_handle, std::initializer_list<std::initializer_list<T>> li):
        MatrixDense(arg_handle, li.size(), (li.size() == 0) ? 0 : std::cbegin(li)->size())
    {

        struct outer_vars {
            int i;
            typename std::initializer_list<std::initializer_list<T>>::const_iterator curr_row;
        };
        struct inner_vars {
            int j;
            typename std::initializer_list<T>::const_iterator curr_elem;
        };

        T *h_mat = static_cast<T *>(malloc(mem_size));

        outer_vars outer = {0, std::cbegin(li)};
        for (; outer.curr_row != std::cend(li); ++outer.curr_row, ++outer.i) {

            inner_vars inner = {0, std::cbegin(*outer.curr_row)};
            for (; inner.curr_elem != std::cend(*outer.curr_row); ++inner.curr_elem, ++inner.j) {

                if (inner.j >= cols()) {
                    free(h_mat);
                    throw(std::runtime_error("Initializer list has non-consistent row size"));
                }
                h_mat[outer.i+inner.j*m] = *inner.curr_elem;

            }

            if (inner.j != cols()) {
                free(h_mat);
                throw(std::runtime_error("Initializer list has non-consistent row size"));
            }

        }

        if (outer.i != rows()) {
            free(h_mat);
            throw(std::runtime_error("Initializer list has non-consistent row size"));
        }

        if ((rows() != 0) && (cols() != 0)) {
            check_cublas_status(cublasSetMatrix(m, n, sizeof(T), h_mat, m, d_mat, m));
        }

        free(h_mat);

    }

    // MatrixDense(const Parent &parent): Parent::Matrix(parent) {}
    // MatrixDense(const Block &block): Parent::Matrix(block.base()) {}
    // MatrixDense(const typename MatrixSparse<T>::Block &block): Parent::Matrix(block.base()) {}

    // *** Destructor ***
    ~MatrixDense() {
        check_cuda_error(cudaFree(d_mat));
        d_mat = nullptr;
    }

    // *** Cast ***
    MatrixSparse<T> sparse() const { return MatrixSparse<T>(Parent::sparseView()); };

    // *** Element Access ***
    const T get_elem(int row, int col) const {
        if ((row < 0) || (row >= m)) { throw std::runtime_error("Invalid matrix row access"); }
        if ((col < 0) || (col >= n)) { throw std::runtime_error("Invalid matrix column access"); }
        T h_elem;
        check_cuda_error(cudaMemcpy(&h_elem, d_mat+row+(col*m), sizeof(T), cudaMemcpyDeviceToHost));
        return h_elem;
    }

    void set_elem(int row, int col, T val) {
        if ((row < 0) || (row >= m)) { throw std::runtime_error("Invalid matrix row access"); }
        if ((col < 0) || (col >= n)) { throw std::runtime_error("Invalid matrix column access"); }
        check_cuda_error(cudaMemcpy(d_mat+row+(col*m), &val, sizeof(T), cudaMemcpyHostToDevice));
    }

    Col col(int _col) { return Parent::col(_col); } 
    Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Properties ***
    int rows() const { return m; }
    int cols() const { return n; }

    void print() {

        T *h_mat = static_cast<T *>(malloc(mem_size));

        check_cublas_status(cublasGetMatrix(m, n, sizeof(T), d_mat, m, h_mat, m));
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                std::cout << static_cast<double>(h_mat[i+j*n]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        free(h_mat);
    
    }

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
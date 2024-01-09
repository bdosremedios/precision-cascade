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
    // friend MatrixSparse<T>;
    // const Parent &base() const { return *this; }

    cublasHandle_t handle;
    int m, n;
    size_t mem_size;
    T *d_mat = nullptr;

    void allocate_d_mat() {
        check_cuda_error(cudaMalloc(&d_mat, mem_size));
    }

    MatrixDense(const cublasHandle_t &arg_handle, T *h_mat, int m_elem, int n_elem):
        MatrixDense(arg_handle, m_elem, n_elem)
    {
        if ((m > 0) && (n > 0)) {
            check_cublas_status(cublasSetMatrix(m, n, sizeof(T), h_mat, m, d_mat, m));
        }
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

                if (inner.j >= n) {
                    free(h_mat);
                    throw(std::runtime_error("Initializer list has non-consistent row size"));
                }
                h_mat[outer.i+inner.j*m] = *inner.curr_elem;

            }

            if (inner.j != n) {
                free(h_mat);
                throw(std::runtime_error("Initializer list has non-consistent row size"));
            }

        }

        if (outer.i != m) {
            free(h_mat);
            throw(std::runtime_error("Initializer list has non-consistent row size"));
        }

        if ((m != 0) && (n != 0)) {
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

    // *** Copy-Assignment ***
    MatrixDense<T> & operator=(const MatrixDense &other) {

        if (this != &other) {

            check_cuda_error(cudaFree(d_mat));
            
            handle = other.handle;
            m = other.m;
            n = other.n;
            mem_size = other.mem_size;

            allocate_d_mat();
            check_cuda_error(cudaMemcpy(d_mat, other.d_mat, mem_size, cudaMemcpyDeviceToDevice));

        }

        return *this;

    }

    // *** Copy-Constructor ***
    MatrixDense(const MatrixDense<T> &other) {
        *this = other;
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

    Col col(int arg_j) { return Col(this, arg_j); }
 
    Block block(int row, int col, int m, int n) { return Block(this, row, col, m, n); }

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
    static MatrixDense<T> Zero(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(0); 
            }
        }
        MatrixDense<T> created_mat(arg_handle, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;

    }

    static MatrixDense<T> Ones(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(1); 
            }
        }
        MatrixDense<T> created_mat(arg_handle, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;
    
    }

    static MatrixDense<T> Identity(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                if (i == j) {
                    h_mat[i+j*arg_m] = static_cast<T>(1);
                } else {
                    h_mat[i+j*arg_m] = static_cast<T>(0);
                }
            }
        }
        MatrixDense<T> created_mat(arg_handle, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;

    }

    static MatrixDense<T> Random(const cublasHandle_t &arg_handle, int arg_m, int arg_n) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(dist(gen)); 
            }
        }
        MatrixDense<T> created_mat(arg_handle, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;
    
    }

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

    // Nested lightweight wrapper class representing matrix column and assignment/elem access
    // Requires: modification by/cast to MatrixVector<T>
    class Col //: private Eigen::Block<Parent, Eigen::Dynamic, 1, true>
    {
    private:

        // using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;
        // using ConstColParent = Eigen::Block<const Parent, Eigen::Dynamic, 1, true>;
        // friend MatrixVector<T>;
        // const ColParent &base() const { return *this; }
        // Col(const ColParent &other): ColParent(other) {}
        // Col(const ConstColParent &other): ColParent(other) {}

        friend MatrixDense<T>;

        const int m;
        const int col_j;
        const MatrixDense<T> *associated_mat_ptr;

        Col(const MatrixDense<T> *arg_associated_mat_ptr, int arg_col_j):
            associated_mat_ptr(arg_associated_mat_ptr), col_j(arg_col_j), m(arg_associated_mat_ptr->m)
        {}

    public:

        Col(const MatrixDense<T>::Col &other): Col(other.associated_mat_ptr, other.col_j) {}

        T get_elem(int arg_i) { return associated_mat_ptr->get_elem(arg_i, col_j); }

        void set_from_vec(const MatrixVector<T> &vec) const {

            if (vec.rows() != m) { throw std::runtime_error("Invalid vector for assignment"); }

            check_cuda_error(
                cudaMemcpy(
                    associated_mat_ptr->d_mat + col_j*m,
                    vec.d_vec,
                    m*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );

        }

        MatrixVector<T> copy_to_vec() const {

            MatrixVector<T> created_vec(associated_mat_ptr->handle, associated_mat_ptr->m);

            check_cuda_error(
                cudaMemcpy(
                    created_vec.d_vec,
                    associated_mat_ptr->d_mat + col_j*m,
                    m*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );

            return created_vec;

        }

    };

    // Nested lightweight wrapper class representing matrix block and assignment/elem access
    // Requires: modification by/cast to MatrixDense<T> and modification by MatrixVector
    class Block //: private Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>
    {
    private:

        // using BlockParent = Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>;
        // friend MatrixDense<T>;
        // const BlockParent &base() const { return *this; }
        // Block(const BlockParent &other): BlockParent(other) {}

        friend MatrixDense<T>;
        const int row_i_start;
        const int col_j_start;
        const int m_rows;
        const int n_cols;
        const MatrixDense<T> *associated_mat_ptr;

        Block(
            const MatrixDense<T> *arg_associated_mat_ptr,
            int arg_row_i_start, int arg_col_j_start,
            int arg_m_rows, int arg_n_cols
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            row_i_start(arg_row_i_start), col_j_start(arg_col_j_start),
            m_rows(arg_m_rows), n_cols(arg_n_cols)
        {}

    public:

        // Block operator=(const MatrixVector<T> vec) { return BlockParent::operator=(vec.base()); }
        // Block operator=(const MatrixDense<T> &mat) { return BlockParent::operator=(mat.base()); }

        Block(const MatrixDense<T>::Block &other):
            Block(
                other.associated_mat_ptr,
                other.row_i_start, other.col_j_start,
                other.m_rows, other.n_cols
            )
        {}

        void set_from_vec(const MatrixVector<T> &vec) const;

        void set_from_mat(const MatrixDense<T> &mat) const;

        MatrixDense<T> copy_to_mat() const;

        T get_elem(int arg_i, int arg_j) { return associated_mat_ptr->get_elem(arg_i, arg_j); }

    };

};

#include "MatrixSparse.h"

#endif
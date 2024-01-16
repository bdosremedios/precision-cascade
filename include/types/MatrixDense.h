#ifndef MATRIXDENSE_H
#define MATRIXDENSE_H

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <random>
#include <initializer_list>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"

#include "MatrixVector.h"

template <typename T> class MatrixSparse;

template <typename T>
class MatrixDense
{
private:

    template <typename> friend class MatrixDense;
    // friend MatrixSparse<T>;

    cublasHandle_t handle;
    int m_rows, n_cols;
    size_t mem_size;
    T *d_mat = nullptr;

    void allocate_d_mat() {
        check_cuda_error(cudaMalloc(&d_mat, mem_size));
    }

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Basic Constructors ***
    MatrixDense(const cublasHandle_t &arg_handle):
        handle(arg_handle), m_rows(0), n_cols(0), mem_size(m_rows*n_cols*sizeof(T))
    { allocate_d_mat(); }

    MatrixDense(const cublasHandle_t &arg_handle, int arg_m, int arg_n):
        handle(arg_handle), m_rows(arg_m), n_cols(arg_n), mem_size(m_rows*n_cols*sizeof(T))
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

            if (outer.curr_row->size() != n_cols) {
                free(h_mat);
                throw(std::runtime_error("MatrixDense: initializer list has non-consistent row size"));
            }

            inner_vars inner = {0, std::cbegin(*outer.curr_row)};
            for (; inner.curr_elem != std::cend(*outer.curr_row); ++inner.curr_elem, ++inner.j) {
                h_mat[outer.i+inner.j*m_rows] = *inner.curr_elem;
            }

        }

        if ((m_rows != 0) && (n_cols != 0)) {
            check_cublas_status(cublasSetMatrix(m_rows, n_cols, sizeof(T), h_mat, m_rows, d_mat, m_rows));
        }

        free(h_mat);

    }

    // *** Dynamic Memory *** (assumes outer code handles dynamic memory properly)
    MatrixDense(const cublasHandle_t &arg_handle, T *h_mat, int m_elem, int n_elem):
        MatrixDense(arg_handle, m_elem, n_elem)
    {
        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasSetMatrix(m_rows, n_cols, sizeof(T), h_mat, m_rows, d_mat, m_rows));
        }
    }

    void copy_data_to_ptr(T *h_mat, int m_elem, int n_elem) const {
        if (m_elem != m_rows) {
            throw std::runtime_error("MatrixDense: invalid m_elem dim for copy_data_to_ptr");
        }
        if (n_elem != n_cols) {
            throw std::runtime_error("MatrixDense: invalid n_elem dim for copy_data_to_ptr");
        }
        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_mat, m_rows));
        }
    }

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
            m_rows = other.m_rows;
            n_cols = other.n_cols;
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

    // *** Element Access ***
    const T get_elem(int row, int col) const {
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("MatrixDense: invalid row access in get_elem");
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in get_elem");
        }
        T h_elem;
        check_cuda_error(cudaMemcpy(&h_elem, d_mat+row+(col*m_rows), sizeof(T), cudaMemcpyDeviceToHost));
        return h_elem;
    }

    void set_elem(int row, int col, T val) {
        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("MatrixDense: invalid row access in set_elem");
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in set_elem");
        }
        check_cuda_error(cudaMemcpy(d_mat+row+(col*m_rows), &val, sizeof(T), cudaMemcpyHostToDevice));
    }

    Col get_col(int col) {
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in col");
        }
        return Col(this, col);
    }
 
    Block get_block(int start_row, int start_col, int block_rows, int block_cols) {
        if ((start_row < 0) || (start_row >= m_rows)) {
            throw std::runtime_error("MatrixDense: invalid starting row in block");
        }
        if ((start_col < 0) || (start_col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid starting col in block");
        }
        if ((block_rows < 0) || (start_row+block_rows > m_rows)) {
            throw std::runtime_error("MatrixDense: invalid number of rows in block");
        }
        if ((block_cols < 0) || (start_col+block_cols > n_cols)) {
            throw std::runtime_error("MatrixDense: invalid number of cols in block");
        }
        return Block(this, start_row, start_col, block_rows, block_cols);
    }

    // *** Properties ***
    int rows() const { return m_rows; }
    int cols() const { return n_cols; }
    cublasHandle_t get_handle() const { return handle; }

    void print() const {

        T *h_mat = static_cast<T *>(malloc(mem_size));

        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_mat, m_rows));
        }

        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                std::cout << static_cast<double>(h_mat[i+j*m_rows]) << " ";
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

    // Needed for testing (don't need to optimize performance)
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

    // *** Substitution *** (correct triangularity assumed)
    MatrixVector<T> back_sub(const MatrixVector<T> &rhs) const {

        if (m_rows != n_cols) {
            throw std::runtime_error("MatrixDense::back_sub: non-square matrix");
        }
        if (m_rows != rhs.rows()) {
            throw std::runtime_error("MatrixDense::back_sub: incompatible matrix and rhs");
        }

        T *h_rhs = static_cast<T *>(malloc(m_rows*sizeof(T)));
        T *h_UT = static_cast<T *>(malloc(m_rows*n_cols*sizeof(T)));
        check_cublas_status(cublasGetVector(m_rows, sizeof(T), rhs.d_vec, 1, h_rhs, 1));
        check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_UT, m_rows));

        for (int col=n_cols-1; col>=0; --col) {
            if (h_UT[col+m_rows*col] != static_cast<T>(0)) {
                h_rhs[col] /= h_UT[col+m_rows*col];
                for (int row=col-1; row>=0; --row) {
                    h_rhs[row] -= h_UT[row+m_rows*col]*h_rhs[col];
                }
            } else {
                throw std::runtime_error("MatrixDense::back_sub: zero diagonal entry encountered in matrix");
            }
        }

        MatrixVector<T> created_vec(handle, h_rhs, m_rows);

        free(h_rhs);
        free(h_UT);

        return created_vec;

    }

    MatrixVector<T> frwd_sub(const MatrixVector<T> &rhs) const {

        if (m_rows != n_cols) {
            throw std::runtime_error("MatrixDense::frwd_sub: non-square matrix");
        }
        if (m_rows != rhs.rows()) {
            throw std::runtime_error("MatrixDense::frwd_sub: incompatible matrix and rhs");
        }

        T *h_rhs = static_cast<T *>(malloc(m_rows*sizeof(T)));
        T *h_LT = static_cast<T *>(malloc(m_rows*n_cols*sizeof(T)));
        check_cublas_status(cublasGetVector(m_rows, sizeof(T), rhs.d_vec, 1, h_rhs, 1));
        check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_LT, m_rows));

        for (int col=0; col<n_cols; ++col) {
            if (h_LT[col+m_rows*col] != static_cast<T>(0)) {
                h_rhs[col] /= h_LT[col+m_rows*col];
                for (int row=col+1; row<m_rows; ++row) {
                    h_rhs[row] -= h_LT[row+m_rows*col]*h_rhs[col];
                }
            } else {
                throw std::runtime_error("MatrixDense::frwd_sub: zero diagonal entry encountered in matrix");
            }
        }

        MatrixVector<T> created_vec(handle, h_rhs, m_rows);

        free(h_rhs);
        free(h_LT);

        return created_vec;

    }

    // *** Cast ***
    // MatrixSparse<T> sparse() const { return MatrixSparse<T>(Parent::sparseView()); };

    template <typename Cast_T>
    MatrixDense<Cast_T> cast() const {

        T *h_mat = static_cast<T *>(malloc(m_rows*n_cols*sizeof(T)));
        Cast_T *h_mat_casted = static_cast<Cast_T *>(malloc(m_rows*n_cols*sizeof(Cast_T)));
        
        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_mat, m_rows));
        }

        for (int j=0; j<n_cols; ++j) {
            for (int i=0; i<m_rows; ++i) {
                h_mat_casted[i+j*m_rows] = static_cast<Cast_T>(h_mat[i+j*m_rows]);
            }
        }

        MatrixDense<Cast_T> created_mat(handle, h_mat_casted, m_rows, n_cols);

        free(h_mat);
        free(h_mat_casted);

        return created_mat;

    }

    // *** Arithmetic and Compound Operations ***
    MatrixDense<T> operator*(const T &scalar) const;
    MatrixDense<T> operator/(const T &scalar) const {
        return operator*(static_cast<T>(1)/scalar);
    }

    MatrixVector<T> operator*(const MatrixVector<T> &vec) const;

    MatrixVector<T> transpose_prod(const MatrixVector<T> &vec) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<T> transpose() const {

        T *h_mat = static_cast<T *>(malloc(m_rows*n_cols*sizeof(T)));
        T *h_mat_trans = static_cast<T *>(malloc(n_cols*m_rows*sizeof(T)));

        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_mat, m_rows));
        }

        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                h_mat_trans[j+i*n_cols] = h_mat[i+j*m_rows];
            }
        }

        MatrixDense<T> created_mat(handle, h_mat_trans, n_cols, m_rows);

        free(h_mat);
        free(h_mat_trans);

        return created_mat;

    }
    
    // Needed for testing (don't need to optimize performance)
    MatrixDense<T> operator*(const MatrixDense<T> &mat) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<T> operator+(const MatrixDense<T> &mat) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<T> operator-(const MatrixDense<T> &mat) const;

    // Needed for testing (don't need to optimize performance)
    T norm() const;

    // Nested lightweight wrapper class representing matrix column and assignment/elem access
    // Requires: modification by/cast to MatrixVector<T>
    class Col
    {
    private:

        friend MatrixDense<T>;

        const int m_rows;
        const int col_idx;
        const MatrixDense<T> *associated_mat_ptr;

        Col(const MatrixDense<T> *arg_associated_mat_ptr, int arg_col_idx):
            associated_mat_ptr(arg_associated_mat_ptr),
            col_idx(arg_col_idx),
            m_rows(arg_associated_mat_ptr->m_rows)
        {}

    public:

        Col(const MatrixDense<T>::Col &other): Col(other.associated_mat_ptr, other.col_idx) {}

        T get_elem(int arg_row) {

            if ((arg_row < 0) || (arg_row >= m_rows)) {
                throw std::runtime_error("MatrixDense::Col: invalid row access in get_elem");
            }

            return associated_mat_ptr->get_elem(arg_row, col_idx);
            
        }

        void set_from_vec(const MatrixVector<T> &vec) const {

            if (vec.rows() != m_rows) {
                throw std::runtime_error("MatrixDense::Col: invalid vector for set_from_vec");
            }

            check_cuda_error(
                cudaMemcpy(
                    associated_mat_ptr->d_mat + col_idx*m_rows,
                    vec.d_vec,
                    m_rows*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );

        }

        MatrixVector<T> copy_to_vec() const {

            MatrixVector<T> created_vec(associated_mat_ptr->handle, associated_mat_ptr->m_rows);

            check_cuda_error(
                cudaMemcpy(
                    created_vec.d_vec,
                    associated_mat_ptr->d_mat + col_idx*m_rows,
                    m_rows*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );

            return created_vec;

        }

    };

    // Nested lightweight wrapper class representing matrix block and assignment/elem access
    // Requires: modification by/cast to MatrixDense<T> and modification by MatrixVector
    class Block
    {
    private:

        friend MatrixDense<T>;
        const int row_idx_start;
        const int col_idx_start;
        const int m_rows;
        const int n_cols;
        const MatrixDense<T> *associated_mat_ptr;

        Block(
            const MatrixDense<T> *arg_associated_mat_ptr,
            int arg_row_idx_start, int arg_col_idx_start,
            int arg_m_rows, int arg_n_cols
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            row_idx_start(arg_row_idx_start), col_idx_start(arg_col_idx_start),
            m_rows(arg_m_rows), n_cols(arg_n_cols)
        {}

    public:

        Block(const MatrixDense<T>::Block &other):
            Block(
                other.associated_mat_ptr,
                other.row_idx_start, other.col_idx_start,
                other.m_rows, other.n_cols
            )
        {}

        void set_from_vec(const MatrixVector<T> &vec) const {

            if (n_cols != 1) {
                throw std::runtime_error("MatrixDense::Block invalid for set_from_vec must be 1 column");
            }
            if (m_rows != vec.rows()) {
                throw std::runtime_error("MatrixDense::Block invalid vector for set_from_vec");
            }

            check_cuda_error(
                cudaMemcpy(
                    (associated_mat_ptr->d_mat +
                     row_idx_start +
                     (col_idx_start*associated_mat_ptr->m_rows)),
                    vec.d_vec,
                    m_rows*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );

        }

        void set_from_mat(const MatrixDense<T> &mat) const {

            if ((m_rows != mat.rows()) || (n_cols != mat.cols())) {
                throw std::runtime_error("MatrixDense::Block invalid matrix for set_from_mat");
            }

            // Copy column by column 1D slices relevant to matrix
            for (int j=0; j<n_cols; ++j) {
                check_cuda_error(
                    cudaMemcpy(
                        (associated_mat_ptr->d_mat +
                         row_idx_start +
                         ((col_idx_start+j)*associated_mat_ptr->m_rows)),
                        mat.d_mat + j*m_rows,
                        m_rows*sizeof(T),
                        cudaMemcpyDeviceToDevice
                    )
                );
            }

        }

        MatrixDense<T> copy_to_mat() const {

            MatrixDense<T> created_mat(associated_mat_ptr->handle, m_rows, n_cols);

            // Copy column by column 1D slices relevant to matrix
            for (int j=0; j<n_cols; ++j) {
                check_cuda_error(
                    cudaMemcpy(
                        created_mat.d_mat + j*m_rows,
                        (associated_mat_ptr->d_mat +
                         row_idx_start +
                         ((col_idx_start+j)*associated_mat_ptr->m_rows)),
                        m_rows*sizeof(T),
                        cudaMemcpyDeviceToDevice
                    )
                );
            }

            return created_mat;

        }

        T get_elem(int row, int col) {

            if ((row < 0) || (row >= m_rows)) {
                throw std::runtime_error("MatrixDense::Block: invalid row access in get_elem");
            }
            if ((col < 0) || (col >= n_cols)) {
                throw std::runtime_error("MatrixDense::Block: invalid col access in get_elem");
            }

            return associated_mat_ptr->get_elem(row_idx_start+row, col_idx_start+col);

        }

    };

};

#include "MatrixSparse.h"

#endif
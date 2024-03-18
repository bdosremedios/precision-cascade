#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H

#include <stdexcept>
#include <cstdlib>
#include <random>
#include <initializer_list>

#include <string>
#include <format>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tools/cuda_check.h"
#include "tools/CuHandleBundle.h"

#include "MatrixDense_gpu_kernels.cuh"

#include "types/Scalar/Scalar.h"

template <typename T> class Scalar;
template <typename T> class Vector;
template <typename T> class MatrixSparse;

template <typename T>
class MatrixDense
{
private:

    template <typename> friend class MatrixDense;
    friend class Vector<T>;
    // friend MatrixSparse<T>;

    cuHandleBundle cu_handles;
    int m_rows = 0, n_cols = 0;
    T *d_mat = nullptr;

    size_t mem_size() const {
        return m_rows*n_cols*sizeof(T);
    }

    void allocate_d_mat() {
        check_cuda_error(cudaMalloc(&d_mat, mem_size()));
    }

    MatrixDense<__half> to_half() const;
    MatrixDense<float> to_float() const;
    MatrixDense<double> to_double() const;

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Constructors ***
    MatrixDense(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ):
        cu_handles(arg_cu_handles),
        m_rows(arg_m_rows),
        n_cols(arg_n_cols)
    { allocate_d_mat(); }

    MatrixDense(const cuHandleBundle &arg_cu_handles): MatrixDense(arg_cu_handles, 0, 0) {}

    // Row-major nested initializer list assumed for intuitive user instantiation
    MatrixDense(
        const cuHandleBundle &arg_cu_handles,
        std::initializer_list<std::initializer_list<T>> li
    ):
        MatrixDense(
            arg_cu_handles,
            li.size(),
            (li.size() == 0) ? 0 : std::cbegin(li)->size()
        )
    {

        struct outer_vars {
            int i;
            typename std::initializer_list<std::initializer_list<T>>::const_iterator curr_row;
        };
        struct inner_vars {
            int j;
            typename std::initializer_list<T>::const_iterator curr_elem;
        };

        T *h_mat = static_cast<T *>(malloc(mem_size()));

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
    MatrixDense(
        const cuHandleBundle &arg_cu_handles,
        T *h_mat,
        int m_elem,
        int n_elem
    ):
        MatrixDense(
            arg_cu_handles,
            m_elem,
            n_elem
        )
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

    // *** Conversion Constructor ***
    MatrixDense(const MatrixDense<T>::Block &block):
        MatrixDense(
            block.associated_mat_ptr->cu_handles,
            block.m_rows,
            block.n_cols
        )
    {
        // Copy column by column 1D slices relevant to matrix
        for (int j=0; j<block.n_cols; ++j) {
            check_cuda_error(
                cudaMemcpy(
                    d_mat + j*block.m_rows,
                    (block.associated_mat_ptr->d_mat +
                        block.row_idx_start +
                        (block.col_idx_start+j)*block.associated_mat_ptr->m_rows),
                    block.m_rows*sizeof(T),
                    cudaMemcpyDeviceToDevice
                )
            );
        }
    }

    // *** Destructor ***
    ~MatrixDense() { check_cuda_error(cudaFree(d_mat)); }

    // *** Copy Assignment ***
    MatrixDense<T> & operator=(const MatrixDense &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;

            if ((m_rows != other.m_rows) || (n_cols != other.n_cols)) {
                check_cuda_error(cudaFree(d_mat));
                m_rows = other.m_rows;
                n_cols = other.n_cols;
                allocate_d_mat();
            }

            check_cuda_error(cudaMemcpy(d_mat, other.d_mat, mem_size(), cudaMemcpyDeviceToDevice));

        }

        return *this;

    }

    // *** Copy Constructor ***
    MatrixDense(const MatrixDense<T> &other) { *this = other; }

    // *** Element Access ***
    const Scalar<T> get_elem(int row, int col) const {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("MatrixDense: invalid row access in get_elem");
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in get_elem");
        }

        Scalar<T> elem;
        check_cuda_error(cudaMemcpy(elem.d_scalar, d_mat+row+(col*m_rows), sizeof(T), cudaMemcpyDeviceToDevice));
        return elem;

    }

    void set_elem(int row, int col, Scalar<T> val) {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("MatrixDense: invalid row access in set_elem");
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in set_elem");
        }

        check_cuda_error(cudaMemcpy(d_mat+row+(col*m_rows), val.d_scalar, sizeof(T), cudaMemcpyDeviceToDevice));

    }

    Col get_col(int col) const {

        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("MatrixDense: invalid col access in col");
        }

        return Col(this, col);

    }
 
    Block get_block(int start_row, int start_col, int block_rows, int block_cols) const {

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
    cuHandleBundle get_cu_handles() const { return cu_handles; }
    int rows() const { return m_rows; }
    int cols() const { return n_cols; }
    int non_zeros() const { 

        T *h_mat = static_cast<T *>(malloc(mem_size()));

        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(m_rows, n_cols, sizeof(T), d_mat, m_rows, h_mat, m_rows));
        }
        
        int count = 0;
        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                if (static_cast<double>(h_mat[i+j*m_rows]) != static_cast<double>(0.)) {
                    count++;
                }
            }
        }

        free(h_mat);

        return count;
    
    }

    std::string get_matrix_string() const {

        T *h_mat = static_cast<T *>(malloc(mem_size()));

        copy_data_to_ptr(h_mat, m_rows, n_cols);

        std::string acc;
        for (int i=0; i<m_rows-1; ++i) {
            for (int j=0; j<n_cols-1; ++j) {
                acc += std::format("{:.6g} ", static_cast<double>(h_mat[i+j*m_rows]));
            }
            acc += std::format("{:.6g}\n", static_cast<double>(h_mat[i+(n_cols-1)*m_rows]));
        }
        for (int j=0; j<n_cols-1; ++j) {
            acc += std::format("{:.6g} ", static_cast<double>(h_mat[(m_rows-1)+j*m_rows]));
        }
        acc += std::format("{:.6g}", static_cast<double>(h_mat[(m_rows-1)+(n_cols-1)*m_rows]));

        free(h_mat);

        return acc;
    
    }

    std::string get_info_string() const {
        int non_zeros_count = non_zeros();
        return std::format(
            "Rows: {} | Cols: {} | Non-zeroes: {} | Fill ratio: {:.3g} | Max magnitude: {:.3g}",
            rows(),
            cols(),
            non_zeros_count,
            static_cast<double>(non_zeros_count)/static_cast<double>(rows()*cols()),
            get_max_mag_elem().get_scalar()
        );
    }

    // *** Static Creation ***
    static MatrixDense<T> Zero(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(0); 
            }
        }
        MatrixDense<T> created_mat(arg_cu_handles, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;

    }

    static MatrixDense<T> Ones(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(1); 
            }
        }
        MatrixDense<T> created_mat(arg_cu_handles, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;
    
    }

    static MatrixDense<T> Identity(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {

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
        MatrixDense<T> created_mat(arg_cu_handles, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;

    }

    // Needed for testing (don't need to optimize performance)
    static MatrixDense<T> Random(const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        T *h_mat = static_cast<T *>(malloc(arg_m*arg_n*sizeof(T)));
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<T>(dist(gen)); 
            }
        }
        MatrixDense<T> created_mat(arg_cu_handles, h_mat, arg_m, arg_n);

        free(h_mat);

        return created_mat;
    
    }

    // *** Resizing ***
    void reduce() { ; } // Do nothing on reduction

    // *** Cast ***
    template <typename Cast_T>
    MatrixDense<Cast_T> cast() const { throw std::runtime_error("MatrixDense: invalid cast conversion"); }

    template <> MatrixDense<__half> cast<__half>() const { return to_half(); }
    template <> MatrixDense<float> cast<float>() const { return to_float(); }
    template <> MatrixDense<double> cast<double>() const { return to_double(); }

    // *** Arithmetic and Compound Operations ***
    MatrixDense<T> operator*(const Scalar<T> &scalar) const;
    MatrixDense<T> operator/(const Scalar<T> &scalar) const {
        Scalar<T> temp(scalar);
        return operator*(temp.reciprocol());
    }
    MatrixDense<T> & operator*=(const Scalar<T> &scalar);
    MatrixDense<T> & operator/=(const Scalar<T> &scalar) {
        Scalar<T> temp(scalar);
        return operator*=(temp.reciprocol());
    }

    Scalar<T> get_max_mag_elem() const {
        
        T *h_mat = static_cast<T *>(malloc(mem_size()));
        copy_data_to_ptr(h_mat, rows(), cols());

        T max_mag = 0.;
        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                T temp = static_cast<T>(std::abs(static_cast<double>(h_mat[i+j*m_rows])));
                if (temp > max_mag) {
                    max_mag = temp;
                }
            }
        }

        free(h_mat);

        return Scalar<T>(max_mag);

    }

    void normalize_magnitude() { *this /= get_max_mag_elem(); }

    Vector<T> operator*(const Vector<T> &vec) const;

    Vector<T> transpose_prod(const Vector<T> &vec) const;

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
        
        MatrixDense<T> created_mat(cu_handles, h_mat_trans, n_cols, m_rows);

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
    Scalar<T> norm() const;

    // *** Substitution *** (correct triangularity assumed)
    Vector<T> back_sub(const Vector<T> &arg_rhs) const;
    Vector<T> frwd_sub(const Vector<T> &arg_rhs) const;

    // Nested lightweight wrapper class representing matrix column and assignment/elem access
    // Requires: modification by/cast to Vector<T>
    class Col
    {
    private:

        friend MatrixDense<T>;
        friend Vector<T>;

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

        Scalar<T> get_elem(int arg_row) {

            if ((arg_row < 0) || (arg_row >= m_rows)) {
                throw std::runtime_error("MatrixDense::Col: invalid row access in get_elem");
            }

            return associated_mat_ptr->get_elem(arg_row, col_idx);

        }

        void set_from_vec(const Vector<T> &vec) const {

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

        Vector<T> copy_to_vec() const {

            Vector<T> created_vec(associated_mat_ptr->get_cu_handles(), associated_mat_ptr->m_rows);

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
    // Requires: modification by/cast to MatrixDense<T> and modification by Vector
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

        void set_from_vec(const Vector<T> &vec) const {

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

            MatrixDense<T> created_mat(associated_mat_ptr->cu_handles, m_rows, n_cols);

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

        Scalar<T> get_elem(int row, int col) {

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

#include "types/Vector/Vector.h"
#include "types/MatrixSparse/ImmutableMatrixSparse.h"

#endif
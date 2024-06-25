#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H

#include "tools/cuda_check.h"
#include "tools/cuHandleBundle.h"
#include "tools/abs.h"
#include "types/Scalar/Scalar.h"
#include "types/Vector/Vector.h"
#include "types/MatrixSparse/NoFillMatrixSparse.h"
#include "MatrixDense_gpu_kernels.cuh"
#include "types/GeneralMatrix/GeneralMatrix_gpu_kernels.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <initializer_list>
#include <cmath>
#include <random>

template <typename TPrecision>
class MatrixDense
{
private:

    template <typename> friend class MatrixDense;
    friend class Vector<TPrecision>;
    friend class NoFillMatrixSparse<TPrecision>;

    cuHandleBundle cu_handles;
    int m_rows = 0;
    int n_cols = 0;
    TPrecision *d_mat = nullptr;

    size_t mem_size() const {
        return m_rows*n_cols*sizeof(TPrecision);
    }

    void allocate_d_mat() {
        check_cuda_error(cudaMalloc(&d_mat, mem_size()));
    }

    MatrixDense<__half> to_half() const;
    MatrixDense<float> to_float() const;
    MatrixDense<double> to_double() const;

    // Use argument overload for type specification rather than explicit
    // specialization due to limitation in g++
    MatrixDense<__half> cast(TypeIdentity<__half> _) const {
        return to_half();
    }
    MatrixDense<float> cast(TypeIdentity<float> _) const {
        return to_float();
    }
    MatrixDense<double> cast(TypeIdentity<double> _) const {
        return to_double();
    }

public:

    class Block; class Col; // Forward declaration of nested classes

    MatrixDense(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ):
        cu_handles(arg_cu_handles),
        m_rows(arg_m_rows),
        n_cols(arg_n_cols)
    {

        if (arg_m_rows < 0) {
            throw std::runtime_error(
                "MatrixDense: invalid arg_m_rows dim for constructor"
            );
        }
        if ((arg_n_cols < 0) || ((arg_m_rows == 0) && (arg_n_cols != 0))) {
            throw std::runtime_error(
                "MatrixDense: invalid arg_n_cols dim for constructor"
            );
        }

        allocate_d_mat();

    }

    MatrixDense(const cuHandleBundle &arg_cu_handles):
        MatrixDense(arg_cu_handles, 0, 0)
    {}

    // Row-major nested initializer list assumed for intuitive user use
    MatrixDense(
        const cuHandleBundle &arg_cu_handles,
        std::initializer_list<std::initializer_list<TPrecision>> li
    ):
        MatrixDense(
            arg_cu_handles,
            li.size(),
            (li.size() == 0) ? 0 : std::cbegin(li)->size()
        )
    {

        struct outer_vars {
            int i;
            typename std::initializer_list<std::initializer_list<TPrecision>>::const_iterator curr_row;
        };
        struct inner_vars {
            int j;
            typename std::initializer_list<TPrecision>::const_iterator curr_elem;
        };

        TPrecision *h_mat = static_cast<TPrecision *>(malloc(mem_size()));

        outer_vars outer = {0, std::cbegin(li)};
        for (; outer.curr_row != std::cend(li); ++outer.curr_row, ++outer.i) {

            if (outer.curr_row->size() != n_cols) {
                free(h_mat);
                throw std::runtime_error(
                    "MatrixDense: initializer list has non-consistent row size"
                );
            }

            inner_vars inner = {0, std::cbegin(*outer.curr_row)};
            for (
                ;
                inner.curr_elem != std::cend(*outer.curr_row);
                ++inner.curr_elem, ++inner.j
            ) {
                h_mat[outer.i+inner.j*m_rows] = *inner.curr_elem;
            }

        }

        if ((m_rows != 0) && (n_cols != 0)) {
            check_cublas_status(cublasSetMatrix(
                m_rows, n_cols, sizeof(TPrecision),
                h_mat, m_rows, d_mat, m_rows
            ));
        }

        free(h_mat);

    }

    MatrixDense(const MatrixDense<TPrecision> &other) {
        *this = other;
    }

    ~MatrixDense() {
        check_cuda_error(cudaFree(d_mat));
    }

    MatrixDense<TPrecision> & operator=(const MatrixDense &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;

            if ((m_rows != other.m_rows) || (n_cols != other.n_cols)) {
                check_cuda_error(cudaFree(d_mat));
                m_rows = other.m_rows;
                n_cols = other.n_cols;
                allocate_d_mat();
            }

            if ((m_rows > 0) && (n_cols > 0)) {
                check_cuda_error(cudaMemcpy(
                    d_mat,
                    other.d_mat,
                    mem_size(),
                    cudaMemcpyDeviceToDevice
                ));
            }

        }

        return *this;

    }

    /* Dynamic Memory Constructor (assumes outer code handles dynamic memory
       properly) */
    MatrixDense(
        const cuHandleBundle &source_cu_handles,
        TPrecision *h_mat,
        int source_m_rows,
        int source_n_cols
    ):
        MatrixDense(
            source_cu_handles,
            source_m_rows,
            source_n_cols
        )
    {
        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasSetMatrix(
                m_rows, n_cols, sizeof(TPrecision),
                h_mat, m_rows, d_mat, m_rows
            ));
        }
    }

    void copy_data_to_ptr(
        TPrecision *h_mat,
        int target_m_rows,
        int target_n_cols
    ) const {

        if (target_m_rows != m_rows) {
            throw std::runtime_error(
                "MatrixDense: invalid target_m_rows dim for copy_data_to_ptr"
            );
        }
        if (target_n_cols != n_cols) {
            throw std::runtime_error(
                "MatrixDense: invalid target_n_cols dim for copy_data_to_ptr"
            );
        }

        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(
                m_rows, n_cols, sizeof(TPrecision),
                d_mat, m_rows, h_mat, m_rows
            ));
        }

    }

    MatrixDense(const NoFillMatrixSparse<TPrecision> &sparse_mat):
        MatrixDense(sparse_mat.get_cu_handles())
    {
        if ((sparse_mat.rows() > 0) && (sparse_mat.cols() > 0)) {
            *this = MatrixDense(
                sparse_mat.get_block(
                    0, 0, sparse_mat.rows(), sparse_mat.cols()
                ).copy_to_mat()
            );
        }
    }

    const Scalar<TPrecision> get_elem(int row, int col) const {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error(
                "MatrixDense: invalid row access in get_elem"
            );
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error(
                "MatrixDense: invalid col access in get_elem"
            );
        }

        Scalar<TPrecision> elem;
        check_cuda_error(cudaMemcpy(
            elem.d_scalar,
            d_mat+row+(col*m_rows),
            sizeof(TPrecision),
            cudaMemcpyDeviceToDevice
        ));
        return elem;

    }

    void set_elem(int row, int col, Scalar<TPrecision> val) {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error(
                "MatrixDense: invalid row access in set_elem"
            );
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error(
                "MatrixDense: invalid col access in set_elem"
            );
        }

        check_cuda_error(cudaMemcpy(
            d_mat+row+(col*m_rows),
            val.d_scalar,
            sizeof(TPrecision),
            cudaMemcpyDeviceToDevice
        ));

    }

    Col get_col(int col) const {

        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error(
                "MatrixDense: invalid col access in get_col"
            );
        }

        return Col(this, col);

    }
 
    Block get_block(
        int start_row, int start_col, int block_rows, int block_cols
    ) const {

        if ((start_row < 0) || (start_row >= m_rows)) {
            throw std::runtime_error(
                "MatrixDense: invalid starting row in block"
            );
        }
        if ((start_col < 0) || (start_col >= n_cols)) {
            throw std::runtime_error(
                "MatrixDense: invalid starting col in block"
            );
        }
        if ((block_rows < 0) || (start_row+block_rows > m_rows)) {
            throw std::runtime_error(
                "MatrixDense: invalid number of rows in block"
            );
        }
        if ((block_cols < 0) || (start_col+block_cols > n_cols)) {
            throw std::runtime_error(
                "MatrixDense: invalid number of cols in block"
            );
        }

        return Block(this, start_row, start_col, block_rows, block_cols);

    }

    cuHandleBundle get_cu_handles() const { return cu_handles; }
    int rows() const { return m_rows; }
    int cols() const { return n_cols; }
    int non_zeros() const {
        
        TPrecision *h_mat = static_cast<TPrecision *>(malloc(mem_size()));
        copy_data_to_ptr(h_mat, m_rows, n_cols);

        int nnz = 0;
        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                if (h_mat[i+j*m_rows] != static_cast<TPrecision>(0.)) {
                    nnz++;
                }
            }
        }

        free(h_mat);

        return nnz;
    }

    std::string get_matrix_string() const {

        TPrecision *h_mat = static_cast<TPrecision *>(malloc(mem_size()));

        copy_data_to_ptr(h_mat, m_rows, n_cols);

        std::stringstream acc_strm;
        acc_strm << std::setprecision(8);
        for (int i=0; i<m_rows-1; ++i) {
            for (int j=0; j<n_cols-1; ++j) {
                acc_strm << static_cast<double>(h_mat[i+j*m_rows]) << " ";
            }
            acc_strm << static_cast<double>(h_mat[i+(n_cols-1)*m_rows]) << "\n";
        }
        for (int j=0; j<n_cols-1; ++j) {
            acc_strm << static_cast<double>(h_mat[(m_rows-1)+j*m_rows]) << " ";
        }
        acc_strm << static_cast<double>(h_mat[(m_rows-1)+(n_cols-1)*m_rows]);

        free(h_mat);

        return acc_strm.str();
    
    }

    std::string get_info_string() const {
        int non_zeros_count = non_zeros();
        std::stringstream acc_strm;
        acc_strm << std::setprecision(3);
        acc_strm << "Rows: " << m_rows << " | "
                 << "Cols: " << n_cols << " | "
                 << "Non-zeroes: " << non_zeros_count << " | "
                 << "Fill ratio: "
                 << (static_cast<double>(non_zeros_count) /
                     static_cast<double>(m_rows*n_cols)) << " | "
                 << "Max magnitude: "
                 << static_cast<double>(get_max_mag_elem().get_scalar());
        return acc_strm.str();
    }

    // Static Matrix Creation
    static MatrixDense<TPrecision> Zero(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<TPrecision>(0); 
            }
        }

        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;

    }

    static MatrixDense<TPrecision> Ones(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<TPrecision>(1); 
            }
        }
        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;
    
    }

    static MatrixDense<TPrecision> Identity(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                if (i == j) {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(1);
                } else {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(0);
                }
            }
        }
        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;

    }

    // Needed for testing (don't need to optimize performance)
    static MatrixDense<TPrecision> Random(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                h_mat[i+j*arg_m] = static_cast<TPrecision>(dist(gen)); 
            }
        }
        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;
    
    }

    // Needed for testing (don't need to optimize performance)
    static MatrixDense<TPrecision> Random_UT(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                if (i <= j) {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(dist(gen));
                    if (i == j) { // Re-roll out of zeros on the diagonal
                        while(h_mat[i+j*arg_m] == static_cast<TPrecision>(0.)) {
                            h_mat[i+j*arg_m] = static_cast<TPrecision>(
                                dist(gen)
                            );
                        }
                    }
                } else {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(0.);
                }
            }
        }

        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;
    
    }

    // Needed for testing (don't need to optimize performance)
    static MatrixDense<TPrecision> Random_LT(
        const cuHandleBundle &arg_cu_handles, int arg_m, int arg_n
    ) {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1., 1.);

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(arg_m*arg_n*sizeof(TPrecision))
        );
        
        for (int j=0; j<arg_n; ++j) {
            for (int i=0; i<arg_m; ++i) {
                if (i >= j) {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(dist(gen));
                    if (i == j) { // Re-roll out of zeros on the diagonal
                        while(h_mat[i+j*arg_m] == static_cast<TPrecision>(0.)) {
                            h_mat[i+j*arg_m] = static_cast<TPrecision>(
                                dist(gen)
                            );
                        }
                    }
                } else {
                    h_mat[i+j*arg_m] = static_cast<TPrecision>(0.);
                }
            }
        }
        MatrixDense<TPrecision> created_mat(
            arg_cu_handles, h_mat, arg_m, arg_n
        );

        free(h_mat);

        return created_mat;
    
    }

    template <typename Cast_TPrecision>
    MatrixDense<Cast_TPrecision> cast() const {
        return cast(TypeIdentity<Cast_TPrecision>());
    }

    // *** Arithmetic and Compound Operations ***
    MatrixDense<TPrecision> operator*(const Scalar<TPrecision> &scalar) const;
    MatrixDense<TPrecision> operator/(const Scalar<TPrecision> &scalar) const {
        Scalar<TPrecision> temp(scalar);
        return operator*(temp.reciprocol());
    }
    MatrixDense<TPrecision> & operator*=(const Scalar<TPrecision> &scalar);
    MatrixDense<TPrecision> & operator/=(const Scalar<TPrecision> &scalar) {
        Scalar<TPrecision> temp(scalar);
        return operator*=(temp.reciprocol());
    }

    Scalar<TPrecision> get_max_mag_elem() const {
        
        TPrecision *h_mat = static_cast<TPrecision *>(malloc(mem_size()));
        copy_data_to_ptr(h_mat, rows(), cols());

        TPrecision max_mag = static_cast<TPrecision>(0.);
        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                TPrecision temp = abs_ns::abs(h_mat[i+j*m_rows]);
                if (temp > max_mag) {
                    max_mag = temp;
                }
            }
        }

        free(h_mat);

        return Scalar<TPrecision>(max_mag);

    }

    void normalize_magnitude() {
        *this /= get_max_mag_elem();
    }

    Vector<TPrecision> operator*(
        const Vector<TPrecision> &vec
    ) const;
    Vector<TPrecision> mult_subset_cols(
        int start, int cols, const Vector<TPrecision> &vec
    ) const;
    Vector<TPrecision> transpose_prod(
        const Vector<TPrecision> &vec
    ) const;
    Vector<TPrecision> transpose_prod_subset_cols(
        int start, int cols, const Vector<TPrecision> &vec
    ) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<TPrecision> transpose() const {

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(m_rows*n_cols*sizeof(TPrecision))
        );
        TPrecision *h_mat_trans = static_cast<TPrecision *>(
            malloc(n_cols*m_rows*sizeof(TPrecision))
        );

        if ((m_rows > 0) && (n_cols > 0)) {
            check_cublas_status(cublasGetMatrix(
                m_rows, n_cols, sizeof(TPrecision),
                d_mat, m_rows, h_mat, m_rows
            ));
        }

        for (int i=0; i<m_rows; ++i) {
            for (int j=0; j<n_cols; ++j) {
                h_mat_trans[j+i*n_cols] = h_mat[i+j*m_rows];
            }
        }
        
        MatrixDense<TPrecision> created_mat(
            cu_handles, h_mat_trans, n_cols, m_rows
        );

        free(h_mat);
        free(h_mat_trans);

        return created_mat;

    }
    
    // Needed for testing (don't need to optimize performance)
    MatrixDense<TPrecision> operator*(const MatrixDense<TPrecision> &mat) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<TPrecision> operator+(const MatrixDense<TPrecision> &mat) const;

    // Needed for testing (don't need to optimize performance)
    MatrixDense<TPrecision> operator-(const MatrixDense<TPrecision> &mat) const;

    // Needed for testing (don't need to optimize performance)
    Scalar<TPrecision> norm() const;

    // Correct triangularity assumed
    Vector<TPrecision> back_sub(const Vector<TPrecision> &arg_rhs) const;
    Vector<TPrecision> frwd_sub(const Vector<TPrecision> &arg_rhs) const;

    /* Nested lightweight wrapper class representing matrix column and
       assignment/elem access
       Requires: modification by/cast to Vector<TPrecision>*/
    class Col
    {
    private:

        friend MatrixDense<TPrecision>;
        friend Vector<TPrecision>;

        const int m_rows;
        const int col_idx;
        const MatrixDense<TPrecision> *associated_mat_ptr;

        Col(
            const MatrixDense<TPrecision> *arg_associated_mat_ptr,
            int arg_col_idx
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            col_idx(arg_col_idx),
            m_rows(arg_associated_mat_ptr->m_rows)
        {}

    public:

        Col(const MatrixDense<TPrecision>::Col &other):
            Col(other.associated_mat_ptr, other.col_idx)
        {}

        Scalar<TPrecision> get_elem(int arg_row) {

            if ((arg_row < 0) || (arg_row >= m_rows)) {
                throw std::runtime_error(
                    "MatrixDense::Col: invalid row access in get_elem"
                );
            }

            return associated_mat_ptr->get_elem(arg_row, col_idx);

        }

        void set_from_vec(const Vector<TPrecision> &vec) const {

            if (vec.rows() != m_rows) {
                throw std::runtime_error(
                    "MatrixDense::Col: invalid vector for set_from_vec"
                );
            }

            check_cuda_error(
                cudaMemcpy(
                    associated_mat_ptr->d_mat + col_idx*m_rows,
                    vec.d_vec,
                    m_rows*sizeof(TPrecision),
                    cudaMemcpyDeviceToDevice
                )
            );

        }

        Vector<TPrecision> copy_to_vec() const {

            Vector<TPrecision> created_vec(
                associated_mat_ptr->get_cu_handles(),
                associated_mat_ptr->m_rows
            );

            check_cuda_error(
                cudaMemcpy(
                    created_vec.d_vec,
                    associated_mat_ptr->d_mat + col_idx*m_rows,
                    m_rows*sizeof(TPrecision),
                    cudaMemcpyDeviceToDevice
                )
            );

            return created_vec;

        }

    };

    /* Nested lightweight wrapper class representing matrix block and
       assignment/elem access
       Requires: modification by/cast to MatrixDense<TPrecision> and
                 modification by Vector */
    class Block
    {
    private:

        friend MatrixDense<TPrecision>;

        const int row_idx_start;
        const int col_idx_start;
        const int m_rows;
        const int n_cols;
        const MatrixDense<TPrecision> *associated_mat_ptr;

        Block(
            const MatrixDense<TPrecision> *arg_associated_mat_ptr,
            int arg_row_idx_start, int arg_col_idx_start,
            int arg_m_rows, int arg_n_cols
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            row_idx_start(arg_row_idx_start), col_idx_start(arg_col_idx_start),
            m_rows(arg_m_rows), n_cols(arg_n_cols)
        {}

    public:

        Block(const MatrixDense<TPrecision>::Block &other):
            Block(
                other.associated_mat_ptr,
                other.row_idx_start, other.col_idx_start,
                other.m_rows, other.n_cols
            )
        {}

        void set_from_vec(const Vector<TPrecision> &vec) const {

            if (n_cols != 1) {
                throw std::runtime_error(
                    "MatrixDense::Block invalid for set_from_vec must be 1 "
                    "column"
                );
            }
            if (m_rows != vec.rows()) {
                throw std::runtime_error(
                    "MatrixDense::Block invalid vector "
                    "for set_from_vec"
                );
            }

            check_cuda_error(
                cudaMemcpy(
                    (associated_mat_ptr->d_mat +
                     row_idx_start +
                     (col_idx_start*associated_mat_ptr->m_rows)),
                    vec.d_vec,
                    m_rows*sizeof(TPrecision),
                    cudaMemcpyDeviceToDevice
                )
            );

        }

        void set_from_mat(const MatrixDense<TPrecision> &mat) const {

            if ((m_rows != mat.rows()) || (n_cols != mat.cols())) {
                throw std::runtime_error(
                    "MatrixDense::Block invalid matrix for set_from_mat"
                );
            }

            // Copy column by column 1D slices relevant to matrix
            for (int j=0; j<n_cols; ++j) {
                check_cuda_error(
                    cudaMemcpy(
                        (associated_mat_ptr->d_mat +
                         row_idx_start +
                         ((col_idx_start+j)*associated_mat_ptr->m_rows)),
                        mat.d_mat + j*m_rows,
                        m_rows*sizeof(TPrecision),
                        cudaMemcpyDeviceToDevice
                    )
                );
            }

        }

        MatrixDense<TPrecision> copy_to_mat() const {

            MatrixDense<TPrecision> created_mat(
                associated_mat_ptr->cu_handles, m_rows, n_cols
            );

            // Copy column by column 1D slices relevant to matrix
            for (int j=0; j<n_cols; ++j) {
                check_cuda_error(
                    cudaMemcpy(
                        created_mat.d_mat + j*m_rows,
                        (associated_mat_ptr->d_mat +
                         row_idx_start +
                         ((col_idx_start+j)*associated_mat_ptr->m_rows)),
                        m_rows*sizeof(TPrecision),
                        cudaMemcpyDeviceToDevice
                    )
                );
            }

            return created_mat;

        }

        Scalar<TPrecision> get_elem(int row, int col) {

            if ((row < 0) || (row >= m_rows)) {
                throw std::runtime_error(
                    "MatrixDense::Block: invalid row access in get_elem"
                );
            }
            if ((col < 0) || (col >= n_cols)) {
                throw std::runtime_error(
                    "MatrixDense::Block: invalid col access in get_elem"
                );
            }

            return associated_mat_ptr->get_elem(
                row_idx_start+row, col_idx_start+col
            );

        }

    };

};

#endif
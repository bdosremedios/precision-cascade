#ifndef NOFILLMATRIXSPARSE_H
#define NOFILLMATRIXSPARSE_H

#include "tools/cuda_check.h"
#include "tools/cuHandleBundle.h"
#include "tools/abs.h"
#include "types/Scalar/Scalar.h"
#include "types/Vector/Vector.h"
#include "types/MatrixDense/MatrixDense.h"
#include "NoFillMatrixSparse_gpu_kernels.cuh"
#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"
#include "types/GeneralMatrix/GeneralMatrix_gpu_kernels.cuh"

#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <random>
#include <vector>

namespace cascade {

// Sparse matrix which disallows fill changes or invidual element changes
// and assumes interfacing with CSC style storage for sake of intuitive uasge.
// Internal implementation CSR due to quirk of CUDA algorithms being better
// tuned for CSR over CSC, however no CSR to CSC transformation is allowed
// for non construction methods (i.e. transform between CSC/CSR is allowed once)
template <typename TPrecision>
class NoFillMatrixSparse
{
private:

    template <typename> friend class NoFillMatrixSparse;

    cuHandleBundle cu_handles;
    int m_rows = 0;
    int n_cols = 0;
    int nnz = 0;
    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    TPrecision *d_values = nullptr;

    std::vector<int> trsv_level_set_cnt;
    std::vector<int *> trsv_level_set_ptrs;

    size_t mem_size_row_offsets() const {
        return (m_rows+1)*sizeof(int);
    }

    size_t mem_size_col_indices() const {
        return nnz*sizeof(int);
    }

    size_t mem_size_values() const {
        return nnz*sizeof(TPrecision);
    }

    size_t mem_size_trsv_preprocess() const {
        return m_rows*sizeof(int);
    }

    void allocate_d_row_offsets() {
        check_cuda_error(cudaMalloc(&d_row_offsets, mem_size_row_offsets()));
    }

    void allocate_d_col_indices() {
        check_cuda_error(cudaMalloc(&d_col_indices, mem_size_col_indices()));
    }

    void allocate_d_values() {
        check_cuda_error(cudaMalloc(&d_values, mem_size_values()));
    }

    void allocate_d_mem() {
        allocate_d_row_offsets();
        allocate_d_col_indices();
        allocate_d_values();
    }

    int binary_search_for_target_index(
        int target, int *arr, int start, int end
    ) const {

        if (start >= end) {
            return -1;
        } else if (start == (end-1)) {
            if (arr[start] == target) {
                return start;
            } else {
                return -1;
            }
        } else {
            int cand_ind = start+(end-start)/2;
            if (arr[cand_ind] == target) {
                return cand_ind;
            } else if (arr[cand_ind] < target) {
                return binary_search_for_target_index(
                    target, arr, cand_ind+1, end
                );
            } else {
                return binary_search_for_target_index(
                    target, arr, start, cand_ind
                );
            }
        }

    }

    NoFillMatrixSparse<__half> to_half() const;
    NoFillMatrixSparse<float> to_float() const;
    NoFillMatrixSparse<double> to_double() const;
    
    // Use argument overload for type specification rather than explicit
    // specialization due to limitation in g++
    NoFillMatrixSparse<__half> cast(TypeIdentity<__half> _) const {
        return to_half();
    }
    NoFillMatrixSparse<float> cast(TypeIdentity<float> _) const {
        return to_float();
    }
    NoFillMatrixSparse<double> cast(TypeIdentity<double> _) const {
        return to_double();
    }

    void set_data_from_csc(
        int *arg_col_offsets, int *arg_row_indices, TPrecision *arg_values,
        int arg_m_rows, int arg_n_cols, int arg_nnz
    ) {

        if (
            (arg_m_rows != m_rows) || (arg_n_cols != n_cols) || (arg_nnz != nnz)
        ) {
            throw std::runtime_error(
                "set_data_from_csc: mismatch in dimensions"
            );
        }

        int *h_row_offsets = static_cast<int *>(malloc(mem_size_row_offsets()));
        int *h_col_indices = static_cast<int *>(malloc(mem_size_col_indices()));
        TPrecision *h_values = static_cast<TPrecision *>(
            malloc(mem_size_values())
        );

        // Calculate row offsets
        for (int j=0; j<(m_rows+1); ++j) { h_row_offsets[j] = 0; }
        for (int k=0; k<nnz; ++k) { ++h_row_offsets[arg_row_indices[k]+1]; }
        for (int j=1; j<(m_rows+1); ++j) {
            h_row_offsets[j] += h_row_offsets[j-1];
        }

        // Fill in column by column col index and value data, with quick offset
        // calculation by using a tracking of the size of a row currently
        // added at any given time
        int row;
        TPrecision val;
        std::vector<int> curr_row_count(m_rows, 0);
        for (int j=0; j<n_cols; ++j) {
            for (
                int offset = arg_col_offsets[j];
                offset < arg_col_offsets[j+1];
                ++offset
            ) {

                row = arg_row_indices[offset];
                val = arg_values[offset];
                
                int new_offset = h_row_offsets[row] + curr_row_count[row];

                h_col_indices[new_offset] = j;
                h_values[new_offset] = val;

                ++curr_row_count[row];

            }
        }

        check_cuda_error(cudaMemcpy(
            d_row_offsets,
            h_row_offsets,
            mem_size_row_offsets(),
            cudaMemcpyHostToDevice
        ));

        if (nnz > 0) {

            check_cuda_error(cudaMemcpy(
                d_col_indices,
                h_col_indices,
                mem_size_col_indices(),
                cudaMemcpyHostToDevice
            ));

            check_cuda_error(cudaMemcpy(
                d_values,
                h_values,
                mem_size_values(),
                cudaMemcpyHostToDevice
            ));

        }

        free(h_row_offsets);
        free(h_col_indices);
        free(h_values);

    }

    void copy_data_to_csr(
        int *h_row_offsets, int *h_col_indices, TPrecision *h_values,
        int target_m_rows, int target_n_cols, int target_nnz
    ) const {

        if (target_m_rows != m_rows) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_m_rows dim for "
                "copy_data_to_csr"
            );
        }
        if (target_n_cols != n_cols) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_n_cols dim for "
                "copy_data_to_csr"
            );
        }
        if (target_nnz != nnz) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_nnz dim for "
                "copy_data_to_csr"
            );
        }

        if (n_cols > 0) {
            check_cuda_error(cudaMemcpy(
                h_row_offsets,
                d_row_offsets,
                mem_size_row_offsets(),
                cudaMemcpyDeviceToHost
            ));
        }

        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                h_col_indices,
                d_col_indices,
                mem_size_col_indices(),
                cudaMemcpyDeviceToHost
            ));
            check_cuda_error(cudaMemcpy(
                h_values,
                d_values,
                mem_size_values(),
                cudaMemcpyDeviceToHost
            ));
        }

    }

    /* Private constructor creating load space for arg_nnz non-zeros but
       without instantiation for use with known sized val array but not known
       values on construction */
    NoFillMatrixSparse(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols,
        int arg_nnz
    ):
        cu_handles(arg_cu_handles),
        m_rows(arg_m_rows),
        n_cols(arg_n_cols),
        nnz(arg_nnz)
    {
        allocate_d_mem();
    }

    /* Private constructor converting MatrixDense to NoFillMatrixSparse
       using cuda subroutine, but differing on specified datatype for
       specialized template class code resuse */
    NoFillMatrixSparse(
        const MatrixDense<TPrecision> &source_mat, cudaDataType cuda_data_type
    ):
        cu_handles(source_mat.get_cu_handles()),
        m_rows(source_mat.rows()),
        n_cols(source_mat.cols()),
        nnz(0)
    {

        size_t col_mem_size = m_rows*sizeof(TPrecision);
        TPrecision *h_rolling_col = static_cast<TPrecision *>(
            malloc(col_mem_size)
        );

        int *h_col_offsets = static_cast<int *>(malloc((n_cols+1)*sizeof(int)));
        std::vector<int> vec_row_indices;
        std::vector<TPrecision> vec_values;

        int nnz_count_so_far = 0;
        for (int j=0; j<n_cols; ++j) {

            h_col_offsets[j] = nnz_count_so_far;

            check_cuda_error(cudaMemcpy(
                h_rolling_col,
                source_mat.d_mat+j*m_rows,
                col_mem_size,
                cudaMemcpyDeviceToHost
            ));

            TPrecision val;
            for (int i=0; i<m_rows; ++i) {
                val = h_rolling_col[i];
                if (val != static_cast<TPrecision>(0.)) {
                    vec_row_indices.push_back(i);
                    vec_values.push_back(val);
                    ++nnz_count_so_far;
                }
            }

        }
        h_col_offsets[n_cols] = nnz_count_so_far;
        nnz = nnz_count_so_far;

        allocate_d_mem();

        if (nnz > 0) {
            set_data_from_csc(
                h_col_offsets, &vec_row_indices[0], &vec_values[0],
                m_rows, n_cols, nnz
            );
        } else {
            set_data_from_csc(
                h_col_offsets, nullptr, nullptr,
                m_rows, n_cols, nnz
            );
        }

        free(h_rolling_col);
        free(h_col_offsets);

    }

    void delete_trsv_preprocess() {
        trsv_level_set_cnt.resize(0);
        for (int *d_lvl_set_ptr : trsv_level_set_ptrs) {
            check_cuda_error(cudaFree(d_lvl_set_ptr));
        };
        trsv_level_set_ptrs.resize(0);
    }

    template <typename WPrecision>
    void deep_copy_trsv_preprocess(
        const NoFillMatrixSparse<WPrecision> &other
    ) {

        // Free current level set preprocessing and deep copy other matrix's
        delete_trsv_preprocess();

        trsv_level_set_cnt = other.trsv_level_set_cnt;
        trsv_level_set_ptrs.resize(other.trsv_level_set_ptrs.size());

        for (int k=0; k < other.trsv_level_set_ptrs.size(); ++k) {
            check_cuda_error(cudaMalloc(
                &(trsv_level_set_ptrs[k]),
                trsv_level_set_cnt[k]*sizeof(int)
            ));
            check_cuda_error(cudaMemcpy(
                trsv_level_set_ptrs[k],
                other.trsv_level_set_ptrs[k],
                trsv_level_set_cnt[k]*sizeof(int),
                cudaMemcpyDeviceToDevice
            ));
        }

    }

    static inline double enlarge_random_double(double val) {
        return val + val/abs_ns::abs(val);
    }

    void check_trsv_dims(const Vector<TPrecision> &soln_rhs) const {

        if ((m_rows != n_cols) || (soln_rhs.rows() != m_rows)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: incorrect dimensions for trsv"
            );
        }

    }

    void slow_back_sub(Vector<TPrecision> &soln_rhs) const;
    void fast_back_sub(Vector<TPrecision> &soln_rhs) const;

    void slow_frwd_sub(Vector<TPrecision> &soln_rhs) const;
    void fast_frwd_sub(Vector<TPrecision> &soln_rhs) const;

public:

    class Block; class Col; // Forward declaration of nested classes

    NoFillMatrixSparse(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ):
        cu_handles(arg_cu_handles),
        m_rows(arg_m_rows),
        n_cols(arg_n_cols),
        nnz(0)
    {

        if (arg_m_rows < 0) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid arg_m_rows dim for constructor"
            );
        }
        if ((arg_n_cols < 0) || ((arg_m_rows == 0) && (arg_n_cols != 0))) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid arg_n_cols dim for constructor"
            );
        }

        allocate_d_mem();

        std::vector<int> h_row_offsets_vec(m_rows+1, 0);

        check_cuda_error(cudaMemcpy(
            d_row_offsets,
            &h_row_offsets_vec[0],
            mem_size_row_offsets(),
            cudaMemcpyHostToDevice
        ));

    }

    NoFillMatrixSparse(const cuHandleBundle &arg_cu_handles):
        NoFillMatrixSparse(arg_cu_handles, 0, 0)
    {}

    NoFillMatrixSparse(
        const cuHandleBundle &arg_cu_handles,
        std::initializer_list<std::initializer_list<TPrecision>> li
    ):
        cu_handles(arg_cu_handles),
        m_rows(li.size()),
        n_cols((li.size() == 0) ? 0 : std::cbegin(li)->size())
    {

        int *h_col_offsets = static_cast<int *>(malloc((n_cols+1)*sizeof(int)));
        std::vector<int> vec_row_indices;
        std::vector<TPrecision> vec_values;

        // Capture iterators for each row
        std::vector<typename std::initializer_list<TPrecision>::const_iterator> row_iters;
        for (
            auto curr_row = std::cbegin(li);
            curr_row != std::cend(li);
            ++curr_row
        ) {
            // Check all row iterators have consistent size before adding
            if (curr_row->size() == n_cols) {
                row_iters.push_back(std::cbegin(*curr_row));
            } else {
                free(h_col_offsets);
                throw std::runtime_error(
                    "NoFillMatrixSparse: Non-consistent row size encounted in "
                    "initializer list"
                );
            }
        }

        /* Roll through iterators repeatedly to access non-zero elements column
           by column calculating offsets based on number of nnz values
           encountered */
        int next_col_offset = 0;
        for (int j=0; j<n_cols; ++j) {

            h_col_offsets[j] = next_col_offset;
            for (int i=0; i<m_rows; ++i) {

                TPrecision val = *(row_iters[i]);
                if (val != static_cast<TPrecision>(0.)) {
                    vec_row_indices.push_back(i);
                    vec_values.push_back(val);
                    ++next_col_offset;
                }
                ++row_iters[i];

            }

        }
        h_col_offsets[n_cols] = next_col_offset;
        nnz = vec_values.size();

        allocate_d_mem();

        if (nnz > 0) {
            set_data_from_csc(
                h_col_offsets, &vec_row_indices[0], &vec_values[0],
                m_rows, n_cols, nnz
            );
        } else {
            set_data_from_csc(
                h_col_offsets, nullptr, nullptr,
                m_rows, n_cols, nnz
            );
        }

        free(h_col_offsets);

    }

    NoFillMatrixSparse(const NoFillMatrixSparse &other) {
        *this = other;
    }

    ~NoFillMatrixSparse() {
        check_cuda_error(cudaFree(d_row_offsets));
        check_cuda_error(cudaFree(d_col_indices));
        check_cuda_error(cudaFree(d_values));
        for (int *d_lvl_set_ptr: trsv_level_set_ptrs) {
            check_cuda_error(cudaFree(d_lvl_set_ptr));
        }
    }

    NoFillMatrixSparse & operator=(const NoFillMatrixSparse &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;
            n_cols = other.n_cols;

            if (m_rows != other.m_rows) {
                check_cuda_error(cudaFree(d_row_offsets));
                m_rows = other.m_rows;
                allocate_d_row_offsets();
            }

            if (m_rows > 0) {
                check_cuda_error(cudaMemcpy(
                    d_row_offsets,
                    other.d_row_offsets,
                    mem_size_row_offsets(),
                    cudaMemcpyDeviceToDevice
                ));
            }

            if (nnz != other.nnz) {
                check_cuda_error(cudaFree(d_col_indices));
                check_cuda_error(cudaFree(d_values));
                nnz = other.nnz;
                allocate_d_col_indices();
                allocate_d_values();
            }

            if (nnz > 0) {
                check_cuda_error(cudaMemcpy(
                    d_col_indices,
                    other.d_col_indices,
                    mem_size_col_indices(),
                    cudaMemcpyDeviceToDevice
                ));
                check_cuda_error(cudaMemcpy(
                    d_values,
                    other.d_values,
                    mem_size_values(),
                    cudaMemcpyDeviceToDevice
                ));
            }

            deep_copy_trsv_preprocess(other);

        }

        return *this;

    }

    NoFillMatrixSparse(const MatrixDense<TPrecision> &source_mat);

    /* Dynamic Memory Constructor (assumes outer code handles dynamic memory
       properly) */
    NoFillMatrixSparse(
        const cuHandleBundle &source_cu_handles,
        int *h_col_offsets,
        int *h_row_indices,
        TPrecision *h_vals,
        int source_m_rows,
        int source_n_cols,
        int source_nnz
    ):
        cu_handles(source_cu_handles),
        m_rows(source_m_rows),
        n_cols(source_n_cols),
        nnz(source_nnz)
    {

        if (source_m_rows < 0) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid source_m_rows dim for dynamic mem "
                "constructor"
            );
        }
        if (source_n_cols < 0) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid source_n_cols dim for dynamic mem "
                "constructor"
            );
        }
        if (source_nnz < 0) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid source_nnz dim for dynamic mem "
                "constructor"
            );
        }

        allocate_d_mem();

        set_data_from_csc(
            h_col_offsets, h_row_indices, h_vals,
            source_m_rows, source_n_cols, source_nnz
        );

    }

    void copy_data_to_ptr(
        int *arg_h_col_offsets,
        int *arg_h_row_indices,
        TPrecision *arg_h_values,
        int target_m_rows, int target_n_cols, int target_nnz
    ) const {

        if (target_m_rows != m_rows) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_m_rows dim for "
                "copy_data_to_ptr"
            );
        }
        if (target_n_cols != n_cols) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_n_cols dim for "
                "copy_data_to_ptr"
            );
        }
        if (target_nnz != nnz) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid target_nnz dim for "
                "copy_data_to_ptr"
            );
        }

        int *h_row_offsets = static_cast<int *>(malloc(mem_size_row_offsets()));
        int *h_col_indices = static_cast<int *>(malloc(mem_size_col_indices()));
        TPrecision *h_values = static_cast<TPrecision *>(
            malloc(mem_size_values())
        );

        copy_data_to_csr(
            h_row_offsets, h_col_indices, h_values,
            m_rows, n_cols, nnz
        );

        // Calculate col offsets
        for (int j=0; j<(n_cols+1); ++j) { arg_h_col_offsets[j] = 0; }
        for (int k=0; k<nnz; ++k) { ++arg_h_col_offsets[h_col_indices[k]+1]; }
        for (int j=1; j<(n_cols+1); ++j) {
            arg_h_col_offsets[j] += arg_h_col_offsets[j-1];
        }

        // Fill in row by row row index and value data, with quick offset
        // calculation by using a tracking of the size of a col currently
        // added at any given time
        int col;
        TPrecision val;
        std::vector<int> curr_col_count(n_cols, 0);
        for (int i=0; i<m_rows; ++i) {
            for (
                int offset = h_row_offsets[i];
                offset < h_row_offsets[i+1];
                ++offset
            ) {

                col = h_col_indices[offset];
                val = h_values[offset];
                
                int new_offset = arg_h_col_offsets[col] + curr_col_count[col];

                arg_h_row_indices[new_offset] = i;
                arg_h_values[new_offset] = val;

                ++curr_col_count[col];

            }
        }

        free(h_row_offsets);
        free(h_col_indices);
        free(h_values);

    }

    const Scalar<TPrecision> get_elem(int row, int col) const {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid row access in get_elem"
            );
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid col access in get_elem"
            );
        }

        // Get column offset and column size
        int row_offset_L;
        int row_offset_R;
        check_cuda_error(cudaMemcpy(
            &row_offset_L,
            d_row_offsets+row,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        check_cuda_error(cudaMemcpy(
            &row_offset_R,
            d_row_offsets+row+1,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        size_t row_nnz_size = row_offset_R-row_offset_L;

        // Find if row index is non-zero and find location in val array
        int *h_col_indices_for_row = static_cast<int *>(
            malloc(row_nnz_size*sizeof(int))
        );
        if (row_nnz_size > 0) {
            check_cuda_error(cudaMemcpy(
                h_col_indices_for_row,
                d_col_indices+row_offset_L,
                row_nnz_size*sizeof(int),
                cudaMemcpyDeviceToHost
            ));
        }
        int col_ind_offset = binary_search_for_target_index(
            col, h_col_indices_for_row, 0, row_nnz_size
        );
        free(h_col_indices_for_row);

        if (col_ind_offset != -1) {
            Scalar<TPrecision> elem;
            check_cuda_error(cudaMemcpy(
                elem.d_scalar,
                d_values+row_offset_L+col_ind_offset,
                sizeof(TPrecision),
                cudaMemcpyDeviceToDevice
            ));
            return elem;
        } else {
            return Scalar<TPrecision>(static_cast<TPrecision>(0.));
        }

    }

    Col get_col(int col) const {

        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid col access in get_col"
            );
        }

        return Col(this, col);

    }

    Block get_block(
        int start_row, int start_col, int block_rows, int block_cols
    ) const {

        if ((start_row < 0) || (start_row >= m_rows)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid starting row in block"
            );
        }
        if ((start_col < 0) || (start_col >= n_cols)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid starting col in block"
            );
        }
        if ((block_rows < 0) || (start_row+block_rows > m_rows)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid number of rows in block"
            );
        }
        if ((block_cols < 0) || (start_col+block_cols > n_cols)) {
            throw std::runtime_error(
                "NoFillMatrixSparse: invalid number of cols in block"
            );
        }

        return Block(this, start_row, start_col, block_rows, block_cols);

    }

    cuHandleBundle get_cu_handles() const { return cu_handles; }
    int rows() const { return m_rows; }
    int cols() const { return n_cols; }
    int non_zeros() const { return nnz; }

    std::string get_matrix_string() const {

        int *h_col_offsets = static_cast<int *>(malloc((n_cols+1)*sizeof(int)));
        int *h_row_indices = static_cast<int *>(malloc(nnz*sizeof(int)));
        TPrecision *h_vals = static_cast<TPrecision *>(
            malloc(nnz*sizeof(TPrecision))
        );

        copy_data_to_ptr(
            h_col_offsets, h_row_indices, h_vals,
            m_rows, n_cols, nnz
        );

        std::stringstream col_offsets_strm;
        col_offsets_strm << "[";
        if (n_cols > 0) {
            for (int i=0; i<n_cols; ++i) {
                col_offsets_strm << h_col_offsets[i] << ", ";
            }
            col_offsets_strm << h_col_offsets[n_cols];
        }
        col_offsets_strm << "]";

        std::stringstream row_indices_strm;
        row_indices_strm << "[";
        std::stringstream val_strm;
        val_strm << std::setprecision(8) << "[";
        if (nnz > 0) {
            for (int i=0; i<nnz-1; ++i) {
                row_indices_strm << h_row_indices[i] << ", ";
                val_strm << static_cast<double>(h_vals[i]) << ", ";
            }
            row_indices_strm << h_row_indices[nnz-1];
            val_strm << static_cast<double>(h_vals[nnz-1]);
        }
        row_indices_strm << "]";
        val_strm << "]";

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return(
            col_offsets_strm.str() + "\n" +
            row_indices_strm.str() + "\n" +
            val_strm.str()
        );

    }

    std::string get_info_string() const {
        std::stringstream acc_strm;
        acc_strm << std::setprecision(3);
        acc_strm << "Rows: " << m_rows << " | "
                << "Cols: " << n_cols << " | "
                << "Non-zeroes: " << nnz << " | "
                << "Fill ratio: "
                << (static_cast<double>(nnz) /
                    static_cast<double>(m_rows*n_cols)) << " | "
                << "Max magnitude: "
                << static_cast<double>(get_max_mag_elem().get_scalar());
        return acc_strm.str();

    }

    // Static Matrix Creation
    static NoFillMatrixSparse<TPrecision> Zero(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ) {
        return NoFillMatrixSparse<TPrecision>(
            arg_cu_handles,
            arg_m_rows,
            arg_n_cols
        );
    }

    static NoFillMatrixSparse<TPrecision> Ones(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ) {

        int *h_col_offsets = static_cast<int *>(
            malloc((arg_n_cols+1)*sizeof(int))
        );
        int *h_row_indices = static_cast<int *>(
            malloc(arg_m_rows*arg_n_cols*sizeof(int))
        );
        TPrecision *h_vals = static_cast<TPrecision *>(
            malloc(arg_m_rows*arg_n_cols*sizeof(TPrecision))
        );

        for (int i=0; i<arg_n_cols+1; ++i) {
            h_col_offsets[i] = i*arg_m_rows;
        }
        for (int i=0; i<arg_m_rows*arg_n_cols; ++i) {
            h_row_indices[i] = i%arg_m_rows;
        }
        for (int i=0; i<arg_m_rows*arg_n_cols; ++i) {
            h_vals[i] = static_cast<TPrecision>(1.);
        }

        NoFillMatrixSparse<TPrecision> ret_mat(
            arg_cu_handles,
            h_col_offsets, h_row_indices, h_vals,
            arg_m_rows, arg_n_cols, arg_m_rows*arg_n_cols
        );

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return ret_mat;

    }

    static NoFillMatrixSparse<TPrecision> Identity(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols
    ) {

        int smaller_dim = std::min(arg_m_rows, arg_n_cols);

        int *h_col_offsets = static_cast<int *>(
            malloc((arg_n_cols+1)*sizeof(int))
        );
        int *h_row_indices = static_cast<int *>(
            malloc(smaller_dim*sizeof(int))
        );
        TPrecision *h_vals = static_cast<TPrecision *>(
            malloc(smaller_dim*sizeof(TPrecision))
        );

        if (arg_n_cols == smaller_dim) {
            for (int i=0; i<arg_n_cols+1; ++i) {
                h_col_offsets[i] = i;
            }
        } else {
            for (int i=0; i<smaller_dim; ++i) {
                h_col_offsets[i] = i;
            }
            for (int i=smaller_dim; i<arg_n_cols+1; ++i) {
                h_col_offsets[i] = smaller_dim;
            }
        }

        for (int i=0; i<smaller_dim; ++i) {
            h_row_indices[i] = i;
        }
    
        for (int i=0; i<smaller_dim; ++i) {
            h_vals[i] = static_cast<TPrecision>(1.);
        }

        NoFillMatrixSparse<TPrecision> ret_mat(
            arg_cu_handles,
            h_col_offsets, h_row_indices, h_vals,
            arg_m_rows, arg_n_cols, smaller_dim
        );

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return ret_mat;

    }

    // Generate a well conditioned pseudo-random matrix with
    // coefficient added to diagonal to avoid illconditioned (don't need to
    // optimize performance)
    static NoFillMatrixSparse<TPrecision> Random(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols,
        double fill_prob
    ) {

        if (!((0. <= fill_prob) && (fill_prob <= 1.))) {
            throw std::runtime_error(
                "NoFillMatrixSparse: Random invalid fill_prob"
            );
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> val_dist(0., 1.);
        std::uniform_real_distribution<double> fill_prob_dist(0., 1.);

        int *h_col_offsets = static_cast<int *>(
            malloc((arg_n_cols+1)*sizeof(int))
        );
        std::vector<int> h_vec_row_indices;
        std::vector<TPrecision> h_vec_vals;

        int curr_nnz = 0;
        for (int j=0; j<arg_n_cols; ++j) {
            h_col_offsets[j] = curr_nnz;
            for (int i=0; i<arg_m_rows; ++i) {

                TPrecision val = static_cast<TPrecision>(0.);

                if (i == j) {
                    val = static_cast<TPrecision>(
                        enlarge_random_double(0.1*val_dist(gen))
                    );
                } else if (
                    (fill_prob != 0.) &&
                    (fill_prob_dist(gen) <= fill_prob)
                ) {
                    val = static_cast<TPrecision>(0.1*val_dist(gen));
                }

                if (val != static_cast<TPrecision>(0.)) {
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                }

            }
        }
        h_col_offsets[arg_n_cols] = curr_nnz;

        NoFillMatrixSparse<TPrecision> created_mat(arg_cu_handles);
        if (curr_nnz != 0) {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                h_col_offsets, &h_vec_row_indices[0], &h_vec_vals[0],
                arg_m_rows, arg_n_cols, curr_nnz
            );
        } else {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                arg_m_rows,
                arg_n_cols
            );
        }

        free(h_col_offsets);

        return created_mat;

    }

    // Generate a well conditioned pseudo-random upper triangular matrix with
    // coefficient added to diagonal to avoid illconditioned (don't need to
    // optimize performance)
    static NoFillMatrixSparse<TPrecision> Random_UT(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols,
        double fill_prob
    ) {

        if (!((0. <= fill_prob) && (fill_prob <= 1.))) {
            throw std::runtime_error(
                "NoFillMatrixSparse: Random invalid fill_prob"
            );
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> val_dist(0., 1.);
        std::uniform_real_distribution<double> fill_prob_dist(0., 1.);

        int *h_col_offsets = static_cast<int *>(malloc((arg_n_cols+1)*sizeof(int)));
        std::vector<int> h_vec_row_indices;
        std::vector<TPrecision> h_vec_vals;

        int curr_nnz = 0;
        for (int j=0; j<arg_n_cols; ++j) {
            h_col_offsets[j] = curr_nnz;
            for (int i=0; ((i <= j) && (i < arg_m_rows)); ++i) {

                TPrecision val = static_cast<TPrecision>(0.);

                if (i == j) {
                    val = static_cast<TPrecision>(
                        enlarge_random_double(0.1*val_dist(gen))
                    );
                } else if (
                    (fill_prob != 0.) &&
                    (fill_prob_dist(gen) <= fill_prob)
                ) {
                    val = static_cast<TPrecision>(0.1*val_dist(gen));
                }

                if (val != static_cast<TPrecision>(0.)) {
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                }

            }
        }
        h_col_offsets[arg_n_cols] = curr_nnz;

        NoFillMatrixSparse<TPrecision> created_mat(arg_cu_handles);
        if (curr_nnz != 0) {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                h_col_offsets, &h_vec_row_indices[0], &h_vec_vals[0],
                arg_m_rows, arg_n_cols, curr_nnz
            );
        } else {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                arg_m_rows,
                arg_n_cols
            );
        }

        free(h_col_offsets);

        return created_mat;

    }

    // Generate a well conditioned pseudo-random lower triangular matrix with
    // coefficient added to diagonal to avoid illconditioned (don't need to
    // optimize performance)
    static NoFillMatrixSparse<TPrecision> Random_LT(
        const cuHandleBundle &arg_cu_handles,
        int arg_m_rows,
        int arg_n_cols,
        double fill_prob
    ) {

        if (!((0. <= fill_prob) && (fill_prob <= 1.))) {
            throw std::runtime_error(
                "NoFillMatrixSparse: Random invalid fill_prob"
            );
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> val_dist(0., 1.);
        std::uniform_real_distribution<double> fill_prob_dist(0., 1.);

        int *h_col_offsets = static_cast<int *>(
            malloc((arg_n_cols+1)*sizeof(int))
        );
        std::vector<int> h_vec_row_indices;
        std::vector<TPrecision> h_vec_vals;

        int curr_nnz = 0;
        for (int j=0; j<arg_n_cols; ++j) {
            h_col_offsets[j] = curr_nnz;
            for (int i=j; i<arg_m_rows; ++i) {

                TPrecision val = static_cast<TPrecision>(0.);

                if (i == j) {
                    val = static_cast<TPrecision>(
                        enlarge_random_double(0.1*val_dist(gen))
                    );
                } else if (
                    (fill_prob != 0.) &&
                    (fill_prob_dist(gen) <= fill_prob)
                ) {
                    val = static_cast<TPrecision>(0.1*val_dist(gen));
                }

                if (val != static_cast<TPrecision>(0.)) {
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                }

            }
        }
        h_col_offsets[arg_n_cols] = curr_nnz;

        NoFillMatrixSparse<TPrecision> created_mat(arg_cu_handles);
        if (curr_nnz != 0) {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                h_col_offsets, &h_vec_row_indices[0], &h_vec_vals[0],
                arg_m_rows, arg_n_cols, curr_nnz
            );
        } else {
            created_mat = NoFillMatrixSparse<TPrecision>(
                arg_cu_handles,
                arg_m_rows,
                arg_n_cols
            );
        }

        free(h_col_offsets);

        return created_mat;

    }

    template <typename Cast_TPrecision>
    NoFillMatrixSparse<Cast_TPrecision> cast() const {
        return cast(TypeIdentity<Cast_TPrecision>());
    }

    NoFillMatrixSparse<TPrecision> operator*(
        const Scalar<TPrecision> &scalar
    ) const;
    NoFillMatrixSparse<TPrecision> operator/(
        const Scalar<TPrecision> &scalar
    ) const {
        Scalar<TPrecision> temp(scalar);
        return operator*(temp.reciprocol());
    }
    
    NoFillMatrixSparse<TPrecision> & operator*=(
        const Scalar<TPrecision> &scalar
    );
    NoFillMatrixSparse<TPrecision> & operator/=(
        const Scalar<TPrecision> &scalar
    ) {
        Scalar<TPrecision> temp(scalar);
        return operator*=(temp.reciprocol());
    }

    Scalar<TPrecision> get_max_mag_elem() const {

        TPrecision *h_values = static_cast<TPrecision *>(
            malloc(mem_size_values())
        );
        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                h_values,
                d_values,
                mem_size_values(),
                cudaMemcpyDeviceToHost
            ));
        }

        TPrecision max_mag = static_cast<TPrecision>(0);
        for (int i=0; i<nnz; ++i) {
            TPrecision temp = abs_ns::abs(h_values[i]);
            if (max_mag < temp) {
                max_mag = temp;
            }
        }

        free(h_values);

        return Scalar<TPrecision>(max_mag);

    }

    void normalize_magnitude() {
        *this /= get_max_mag_elem();
    }

    Vector<TPrecision> operator*(const Vector<TPrecision> &vec) const;

    Vector<TPrecision> transpose_prod(const Vector<TPrecision> &vec) const;

    // Use fact that transpose of CSR matrix is same arrays used as CSC
    NoFillMatrixSparse<TPrecision> transpose() const {

        int *curr_h_row_offsets = static_cast<int *>(
            malloc(mem_size_row_offsets())
        );
        int *curr_h_col_indices = static_cast<int *>(
            malloc(mem_size_col_indices())
        );
        TPrecision *curr_h_values = static_cast<TPrecision *>(
            malloc(mem_size_values())
        );

        copy_data_to_csr(
            curr_h_row_offsets, curr_h_col_indices, curr_h_values,
            m_rows, n_cols, nnz
        );

        NoFillMatrixSparse<TPrecision> created_mat(
            cu_handles,
            curr_h_row_offsets, curr_h_col_indices, curr_h_values,
            n_cols, m_rows, nnz
        );

        free(curr_h_row_offsets);
        free(curr_h_col_indices);
        free(curr_h_values);

        return created_mat;

    }

    bool get_has_fast_trsv() const {
        return !(
            trsv_level_set_cnt.empty() ||
            trsv_level_set_ptrs.empty()
        );
    }

    // Get count of things that can be solved before needing to wait for
    // components in block to be solved
    void preprocess_trsv(bool is_upptri);

    // Allow deletion of preprocess for testing
    void clear_preprocess_trsv() { delete_trsv_preprocess(); }

    Vector<TPrecision> back_sub(const Vector<TPrecision> &arg_rhs) const;
    Vector<TPrecision> frwd_sub(const Vector<TPrecision> &arg_rhs) const;

    /* Nested lightweight wrapper class representing matrix column and
       assignment/elem access
       Requires: cast to Vector<TPrecision> */
    class Col
    {
    private:

        friend NoFillMatrixSparse<TPrecision>;

        const int m_rows;
        const int col_idx;
        const NoFillMatrixSparse<TPrecision> *associated_mat_ptr;

        Col(
            const NoFillMatrixSparse<TPrecision> *arg_associated_mat_ptr,
            int arg_col_idx
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            col_idx(arg_col_idx),
            m_rows(arg_associated_mat_ptr->m_rows)
        {}

    public:

        Col(const NoFillMatrixSparse<TPrecision>::Col &other):
            Col(other.associated_mat_ptr, other.col_idx)
        {}

        Scalar<TPrecision> get_elem(int arg_row) const {

            if ((arg_row < 0) || (arg_row >= m_rows)) {
                throw std::runtime_error(
                    "NoFillMatrixSparse::Col: invalid row access in get_elem"
                );
            }

            return associated_mat_ptr->get_elem(arg_row, col_idx);

        }

        Vector<TPrecision> copy_to_vec() const {

            TPrecision *h_vec = static_cast<TPrecision*>(
                malloc(m_rows*sizeof(TPrecision))
            );

            for (int i=0; i<m_rows; ++i) {
                h_vec[i] = get_elem(i).get_scalar();
            }

            Vector<TPrecision> created_vec(
                associated_mat_ptr->cu_handles, h_vec, m_rows
            );

            free(h_vec);

            return created_vec;

        }

    };

    /* Nested lightweight wrapper class representing matrix block and elem
       access
       Requires: cast to MatrixDense<T> */
    class Block
    {
    private:

        friend NoFillMatrixSparse<TPrecision>;

        const int row_idx_start;
        const int col_idx_start;
        const int m_rows;
        const int n_cols;
        const NoFillMatrixSparse<TPrecision> *associated_mat_ptr;

        Block(
            const NoFillMatrixSparse<TPrecision> *arg_associated_mat_ptr,
            int arg_row_idx_start, int arg_col_idx_start,
            int arg_m_rows, int arg_n_cols
        ):
            associated_mat_ptr(arg_associated_mat_ptr),
            row_idx_start(arg_row_idx_start), col_idx_start(arg_col_idx_start),
            m_rows(arg_m_rows), n_cols(arg_n_cols)
        {}
    
    public:

        Block(const NoFillMatrixSparse<TPrecision>::Block &other):
            Block(
                other.associated_mat_ptr,
                other.row_idx_start, other.col_idx_start,
                other.m_rows, other.n_cols
            )
        {}

        MatrixDense<TPrecision> copy_to_mat() const {

            TPrecision *h_mat = static_cast<TPrecision *>(
                malloc(m_rows*n_cols*sizeof(TPrecision))
            );
            for (int i=0; i<m_rows*n_cols; ++i) {
                h_mat[i] = static_cast<TPrecision>(0.);
            }

            int *h_col_offsets = static_cast<int *>(
                malloc((associated_mat_ptr->n_cols+1)*sizeof(int))
            );
            int *h_row_indices = static_cast<int *>(
                malloc(associated_mat_ptr->nnz*sizeof(int))
            );
            TPrecision *h_vals = static_cast<TPrecision *>(
                malloc(associated_mat_ptr->nnz*sizeof(TPrecision))
            );
            associated_mat_ptr->copy_data_to_ptr(
                h_col_offsets,
                h_row_indices,
                h_vals,
                associated_mat_ptr->m_rows,
                associated_mat_ptr->n_cols,
                associated_mat_ptr->nnz
            );

            // Copy column by column 1D slices relevant to matrix
            for (int j=0; j<n_cols; ++j) {

                /* Get offsets of row indices/values for corresponding offseted
                   column of block */
                int col_offset_L = h_col_offsets[col_idx_start + j];
                int col_offset_R = h_col_offsets[col_idx_start + j + 1];
                int col_size_nnz = col_offset_R-col_offset_L;

                // Load values into h_mat if they are within the block rows
                int cand_row_ind;
                for (int i=0; i<col_size_nnz; ++i) {
                    cand_row_ind = h_row_indices[col_offset_L+i];
                    if (
                        (row_idx_start <= cand_row_ind) &&
                        (cand_row_ind < row_idx_start + m_rows)
                    ) {
                        h_mat[(cand_row_ind-row_idx_start)+j*m_rows] = (
                            h_vals[col_offset_L+i]
                        );
                    }
                }

            }

            MatrixDense<TPrecision> created_mat(
                associated_mat_ptr->cu_handles,
                h_mat,
                m_rows,
                n_cols
            );

            free(h_mat);
            free(h_col_offsets);
            free(h_row_indices);
            free(h_vals);

            return created_mat;

        }

        Scalar<TPrecision> get_elem(int row, int col) const {

            if ((row < 0) || (row >= m_rows)) {
                throw std::runtime_error(
                    "NoFillMatrixSparse::Block: invalid row access in get_elem"
                );
            }
            if ((col < 0) || (col >= n_cols)) {
                throw std::runtime_error(
                    "NoFillMatrixSparse::Block: invalid col access in get_elem"
                );
            }

            return associated_mat_ptr->get_elem(
                row_idx_start+row, col_idx_start+col
            );

        }

    };

};

}

#endif
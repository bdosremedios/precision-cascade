#include "types/Vector/Vector.h"

#ifndef NOFILLMATRIXSPARSE_H
#define NOFILLMATRIXSPARSE_H

#include "tools/cuda_check.h"
#include "tools/cuHandleBundle.h"
#include "types/Scalar/Scalar.h"
#include "types/MatrixDense/MatrixDense.h"
#include "NoFillMatrixSparse_gpu_kernels.cuh"
#include "types/GeneralMatrix/GeneralMatrix_gpu_kernels.cuh"

#include <cmath>
#include <random>
#include <vector>
#include <format>

template <typename TPrecision>
class NoFillMatrixSparse
{
private:

    template <typename> friend class NoFillMatrixSparse;

    cuHandleBundle cu_handles;
    int m_rows = 0;
    int n_cols = 0;
    int nnz = 0;
    int *d_col_offsets = nullptr;
    int *d_row_indices = nullptr;
    TPrecision *d_vals = nullptr;

    size_t mem_size_col_offsets() const {
        return (n_cols+1)*sizeof(int);
    }

    size_t mem_size_row_indices() const {
        return nnz*sizeof(int);
    }

    size_t mem_size_vals() const {
        return nnz*sizeof(TPrecision);
    }

    void allocate_d_col_offsets() {
        check_cuda_error(cudaMalloc(&d_col_offsets, mem_size_col_offsets()));
    }

    void allocate_d_row_indices() {
        check_cuda_error(cudaMalloc(&d_row_indices, mem_size_row_indices()));
    }

    void allocate_d_vals() {
        check_cuda_error(cudaMalloc(&d_vals, mem_size_vals()));
    }

    void allocate_d_mem() {
        allocate_d_col_offsets();
        allocate_d_row_indices();
        allocate_d_vals();
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

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        std::vector<int> vec_row_indices;
        std::vector<TPrecision> vec_vals;

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
                    vec_vals.push_back(val);
                    ++nnz_count_so_far;
                }
            }

        }
        h_col_offsets[n_cols] = nnz_count_so_far;
        nnz = nnz_count_so_far;

        allocate_d_mem();

        check_cuda_error(cudaMemcpy(
            d_col_offsets,
            h_col_offsets,
            mem_size_col_offsets(),
            cudaMemcpyHostToDevice
        ));

        if (nnz > 0) {

            check_cuda_error(cudaMemcpy(
                d_row_indices,
                &vec_row_indices[0],
                mem_size_row_indices(),
                cudaMemcpyHostToDevice
            ));

            check_cuda_error(cudaMemcpy(
                d_vals,
                &vec_vals[0],
                mem_size_vals(),
                cudaMemcpyHostToDevice
            ));

        }

        free(h_rolling_col);
        free(h_col_offsets);

    }

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

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        for (int i=0; i<n_cols+1; ++i) { h_col_offsets[i] = 0; }
        check_cuda_error(cudaMemcpy(
            d_col_offsets,
            h_col_offsets,
            mem_size_col_offsets(),
            cudaMemcpyHostToDevice
        ));
        free(h_col_offsets);

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

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
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

        /* Set remaining host vectors to values to load, and load column
           offsets, row indices, and values into gpu memory */
        allocate_d_mem();
        int *h_row_indices = static_cast<int *>(malloc(mem_size_row_indices()));
        TPrecision *h_vals = static_cast<TPrecision *>(malloc(mem_size_vals()));
        for (int i=0; i<nnz; ++i) { h_row_indices[i] = vec_row_indices[i]; }
        for (int i=0; i<nnz; ++i) { h_vals[i] = vec_values[i]; }

        if (n_cols > 0) {
            check_cuda_error(cudaMemcpy(
                d_col_offsets,
                h_col_offsets,
                mem_size_col_offsets(),
                cudaMemcpyHostToDevice 
            ));
        }

        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                d_row_indices,
                h_row_indices,
                mem_size_row_indices(),
                cudaMemcpyHostToDevice 
            ));
            check_cuda_error(cudaMemcpy(
                d_vals,
                h_vals,
                mem_size_vals(),
                cudaMemcpyHostToDevice 
            ));
        }

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

    }

    NoFillMatrixSparse(const NoFillMatrixSparse &other) {
        *this = other;
    }

    ~NoFillMatrixSparse() {
        check_cuda_error(cudaFree(d_col_offsets));
        check_cuda_error(cudaFree(d_row_indices));
        check_cuda_error(cudaFree(d_vals));
    }

    NoFillMatrixSparse & operator=(const NoFillMatrixSparse &other) {

        if (this != &other) {

            cu_handles = other.cu_handles;
            m_rows = other.m_rows;

            if (n_cols != other.n_cols) {
                check_cuda_error(cudaFree(d_col_offsets));
                n_cols = other.n_cols;
                allocate_d_col_offsets();
            }

            if (n_cols > 0) {
                check_cuda_error(cudaMemcpy(
                    d_col_offsets,
                    other.d_col_offsets,
                    mem_size_col_offsets(),
                    cudaMemcpyDeviceToDevice
                ));
            }

            if (nnz != other.nnz) {
                check_cuda_error(cudaFree(d_row_indices));
                check_cuda_error(cudaFree(d_vals));
                nnz = other.nnz;
                allocate_d_row_indices();
                allocate_d_vals();
            }

            if (nnz > 0) {
                check_cuda_error(cudaMemcpy(
                    d_row_indices,
                    other.d_row_indices,
                    mem_size_row_indices(),
                    cudaMemcpyDeviceToDevice
                ));
                check_cuda_error(cudaMemcpy(
                    d_vals,
                    other.d_vals,
                    mem_size_vals(),
                    cudaMemcpyDeviceToDevice
                ));
            }

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

        if (n_cols > 0) {
            check_cuda_error(cudaMemcpy(
                d_col_offsets,
                h_col_offsets,
                mem_size_col_offsets(),
                cudaMemcpyHostToDevice
            ));
        }

        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                d_row_indices,
                h_row_indices,
                mem_size_row_indices(),
                cudaMemcpyHostToDevice
            ));
            check_cuda_error(cudaMemcpy(
                d_vals,
                h_vals,
                mem_size_vals(),
                cudaMemcpyHostToDevice
            ));
        }

    }

    void copy_data_to_ptr(
        int *h_col_offsets, int *h_row_indices, TPrecision *h_vals,
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

        if (n_cols > 0) {
            check_cuda_error(cudaMemcpy(
                h_col_offsets,
                d_col_offsets,
                mem_size_col_offsets(),
                cudaMemcpyDeviceToHost
            ));
        }

        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                h_row_indices,
                d_row_indices,
                mem_size_row_indices(),
                cudaMemcpyDeviceToHost
            ));
            check_cuda_error(cudaMemcpy(
                h_vals,
                d_vals,
                mem_size_vals(),
                cudaMemcpyDeviceToHost
            ));
        }

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
        int col_offset_L;
        int col_offset_R;
        check_cuda_error(cudaMemcpy(
            &col_offset_L,
            d_col_offsets+col,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        check_cuda_error(cudaMemcpy(
            &col_offset_R,
            d_col_offsets+col+1,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        size_t col_nnz_size = col_offset_R-col_offset_L;

        // Find if row index is non-zero and find location in val array
        int *h_row_indices_for_col = static_cast<int *>(
            malloc(col_nnz_size*sizeof(int))
        );
        if (col_nnz_size > 0) {
            check_cuda_error(cudaMemcpy(
                h_row_indices_for_col,
                d_row_indices+col_offset_L,
                col_nnz_size*sizeof(int),
                cudaMemcpyDeviceToHost
            ));
        }
        int row_ind_offset = binary_search_for_target_index(
            row, h_row_indices_for_col, 0, col_nnz_size
        );
        free(h_row_indices_for_col);

        if (row_ind_offset != -1) {
            Scalar<TPrecision> elem;
            check_cuda_error(cudaMemcpy(
                elem.d_scalar,
                d_vals+col_offset_L+row_ind_offset,
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

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        int *h_row_indices = static_cast<int *>(malloc(mem_size_row_indices()));
        TPrecision *h_vals = static_cast<TPrecision *>(malloc(mem_size_vals()));

        copy_data_to_ptr(
            h_col_offsets, h_row_indices, h_vals,
            m_rows, n_cols, nnz
        );

        std::string col_offsets_str = "[";
        if (n_cols > 0) {
            for (int i=0; i<n_cols; ++i) {
                col_offsets_str += std::format("{}, ", h_col_offsets[i]);
            }
            col_offsets_str += std::format("{}", h_col_offsets[n_cols]);
        }
        col_offsets_str += "]";

        std::string row_indices_str = "[";
        std::string val_str = "[";
        if (nnz > 0) {
            for (int i=0; i<nnz-1; ++i) {
                row_indices_str += std::format("{}, ", h_row_indices[i]);
                val_str += std::format(
                    "{:.6g}, ",
                    static_cast<double>(h_vals[i])
                );
            }
            row_indices_str += std::format("{}", h_row_indices[nnz-1]);
            val_str += std::format("{:.6g}", static_cast<double>(h_vals[nnz-1]));
        }
        row_indices_str += "]";
        val_str += "]";

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return col_offsets_str + "\n" + row_indices_str + "\n" + val_str;

    }

    std::string get_info_string() const {

        return std::format(
            "Rows: {} | Cols: {} | Non-zeroes: {} | Fill ratio: {:.3g} | "
            "Max magnitude: {:.3g}",
            m_rows,
            n_cols,
            nnz,
            static_cast<double>(nnz)/static_cast<double>(m_rows*n_cols),
            static_cast<double>(get_max_mag_elem().get_scalar())
        );

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
        std::uniform_real_distribution<double> val_dist(-1., 1.);
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
                // Enforce diagonal is non-zero for sake of non-singularity
                if (i == j) {
                    TPrecision val = val_dist(gen);
                    while (val == static_cast<TPrecision>(0.)) {
                        val = val_dist(gen);
                    }
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                } else if (
                    (fill_prob != 0.) &&
                    (fill_prob_dist(gen) <= fill_prob)
                ) {
                    TPrecision val = val_dist(gen);
                    if (val != static_cast<TPrecision>(0.)) {
                        h_vec_row_indices.push_back(i);
                        h_vec_vals.push_back(val);
                        ++curr_nnz;
                    }
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
        std::uniform_real_distribution<double> val_dist(-1., 1.);
        std::uniform_real_distribution<double> fill_prob_dist(0., 1.);

        int *h_col_offsets = static_cast<int *>(malloc((arg_n_cols+1)*sizeof(int)));
        std::vector<int> h_vec_row_indices;
        std::vector<TPrecision> h_vec_vals;

        int curr_nnz = 0;
        for (int j=0; j<arg_n_cols; ++j) {
            h_col_offsets[j] = curr_nnz;
            for (int i=0; ((i <= j) && (i < arg_m_rows)); ++i) {
                // Enforce diagonal is non-zero for sake of non-singularity
                if (i == j) {
                    TPrecision val = val_dist(gen);
                    while (val == static_cast<TPrecision>(0.)) {
                        val = val_dist(gen);
                    }
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                } else if ((fill_prob != 0.) && (fill_prob_dist(gen) <= fill_prob)) {
                    TPrecision val = val_dist(gen);
                    if (val != static_cast<TPrecision>(0.)) {
                        h_vec_row_indices.push_back(i);
                        h_vec_vals.push_back(val);
                        ++curr_nnz;
                    }
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
        std::uniform_real_distribution<double> val_dist(-1., 1.);
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
                // Enforce diagonal is non-zero for sake of non-singularity
                if (i == j) {
                    TPrecision val = val_dist(gen);
                    while (val == static_cast<TPrecision>(0.)) {
                        val = val_dist(gen);
                    }
                    h_vec_row_indices.push_back(i);
                    h_vec_vals.push_back(val);
                    ++curr_nnz;
                } else if (
                    (fill_prob != 0.) &&
                    (fill_prob_dist(gen) <= fill_prob)
                ) {
                    TPrecision val = val_dist(gen);
                    if (val != static_cast<TPrecision>(0.)) {
                        h_vec_row_indices.push_back(i);
                        h_vec_vals.push_back(val);
                        ++curr_nnz;
                    }
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

    // *** Explicit Cast ***
    template <typename Cast_T>
    NoFillMatrixSparse<Cast_T> cast() const {
        throw std::runtime_error("NoFillMatrixSparse: invalid cast conversion");
    }

    template <> NoFillMatrixSparse<__half> cast<__half>() const {
        return to_half();
    }
    template <> NoFillMatrixSparse<float> cast<float>() const {
        return to_float();
    }
    template <> NoFillMatrixSparse<double> cast<double>() const {
        return to_double();
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

        TPrecision *h_vals = static_cast<TPrecision *>(malloc(mem_size_vals()));
        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                h_vals,
                d_vals,
                mem_size_vals(),
                cudaMemcpyDeviceToHost
            ));
        }

        TPrecision max_mag = static_cast<TPrecision>(0);
        for (int i=0; i<nnz; ++i) {
            TPrecision temp = static_cast<TPrecision>(
                std::abs(static_cast<double>(h_vals[i]))
            );
            if (max_mag < temp) {
                max_mag = temp;
            }
        }

        free(h_vals);

        return Scalar<TPrecision>(max_mag);

    }

    void normalize_magnitude() {
        *this /= get_max_mag_elem();
    }

    Vector<TPrecision> operator*(const Vector<TPrecision> &vec) const;

    Vector<TPrecision> transpose_prod(const Vector<TPrecision> &vec) const;

    NoFillMatrixSparse<TPrecision> transpose() const {

        int *curr_h_col_offsets = static_cast<int *>(
            malloc(mem_size_col_offsets())
        );
        int *curr_h_row_indices = static_cast<int *>(
            malloc(mem_size_row_indices())
        );
        TPrecision *curr_h_vals = static_cast<TPrecision *>(
            malloc(mem_size_vals())
        );

        copy_data_to_ptr(
            curr_h_col_offsets, curr_h_row_indices, curr_h_vals,
            m_rows, n_cols, nnz
        );

        int *trans_h_col_offsets = static_cast<int *>(
            malloc((m_rows+1)*sizeof(int))
        );
        int *trans_h_row_indices = static_cast<int *>(
            malloc(mem_size_row_indices())
        );
        TPrecision *trans_h_vals = static_cast<TPrecision *>(
            malloc(mem_size_vals())
        );

        for (int j=0; j<m_rows+1; ++j) {
            trans_h_col_offsets[j] = 0;
        }
        for (int k=0; k<nnz; ++k) {
            ++trans_h_col_offsets[curr_h_row_indices[k]+1];
        }
        for (int j=1; j<m_rows+1; ++j) {
            trans_h_col_offsets[j] += trans_h_col_offsets[j-1];
        }

        int *trans_col_count = static_cast<int *>(malloc(m_rows*sizeof(int)));
        for (int j=0; j<m_rows; ++j) {
            trans_col_count[j] = 0;
        }

        int curr_col_ind = 0;
        for (int k=0; k<nnz; ++k) {

            // Ensure we are we have the correct current column index
            while (
                (curr_col_ind < n_cols-1) &&
                (k >= curr_h_col_offsets[curr_col_ind+1])
            ) {
                ++curr_col_ind;
            }
            int curr_row_ind = curr_h_row_indices[k];
            TPrecision curr_val = curr_h_vals[k];

            int trans_k_location = (
                trans_h_col_offsets[curr_row_ind] +
                trans_col_count[curr_row_ind]
            );
            trans_h_row_indices[trans_k_location] = curr_col_ind;
            trans_h_vals[trans_k_location] = curr_val;

            ++trans_col_count[curr_row_ind];

        }

        free(trans_col_count);

        free(curr_h_col_offsets);
        free(curr_h_row_indices);
        free(curr_h_vals);

        NoFillMatrixSparse<TPrecision> created_mat(
            cu_handles,
            trans_h_col_offsets, trans_h_row_indices, trans_h_vals,
            n_cols, m_rows, nnz
        );

        free(trans_h_col_offsets);
        free(trans_h_row_indices);
        free(trans_h_vals);

        return created_mat;

    }

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

        Scalar<TPrecision> get_elem(int arg_row) {

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
                h_vec[i] = static_cast<TPrecision>(0.);
            }

            // Get column offset and column size
            int col_offset_L;
            int col_offset_R;
            check_cuda_error(cudaMemcpy(
                &col_offset_L,
                associated_mat_ptr->d_col_offsets + col_idx,
                sizeof(int),
                cudaMemcpyDeviceToHost
            ));
            check_cuda_error(cudaMemcpy(
                &col_offset_R,
                associated_mat_ptr->d_col_offsets + col_idx + 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            ));
            size_t col_nnz_size = col_offset_R-col_offset_L;

            int *h_row_indices = static_cast<int *>(
                malloc(col_nnz_size*sizeof(int))
            );
            TPrecision *h_vals = static_cast<TPrecision *>(
                malloc(col_nnz_size*sizeof(TPrecision))
            );
            if (col_nnz_size > 0) {
                check_cuda_error(cudaMemcpy(
                    h_row_indices,
                    associated_mat_ptr->d_row_indices + col_offset_L,
                    col_nnz_size*sizeof(int),
                    cudaMemcpyDeviceToHost
                ));
                check_cuda_error(cudaMemcpy(
                    h_vals,
                    associated_mat_ptr->d_vals + col_offset_L,
                    col_nnz_size*sizeof(TPrecision),
                    cudaMemcpyDeviceToHost
                ));
            }

            for (int i=0; i<col_nnz_size; ++i) {
                h_vec[h_row_indices[i]] = h_vals[i];
            }

            Vector<TPrecision> created_vec(
                associated_mat_ptr->cu_handles, h_vec, m_rows
            );

            free(h_vec);
            free(h_row_indices);
            free(h_vals);

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

        Scalar<TPrecision> get_elem(int row, int col) {

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

#endif
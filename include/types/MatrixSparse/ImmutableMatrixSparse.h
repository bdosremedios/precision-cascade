#ifndef IMMUTABLEMATRIXSPARSE_H
#define IMMUTABLEMATRIXSPARSE_H

// #include "MatrixVector.h"
// #include "MatrixDense.h"

#include <cmath>

#include "tools/cuda_check.h"
#include "tools/cuHandleBundle.h"

template <typename T>
class ImmutableMatrixSparse
{
private:

    cuHandleBundle cu_handles;
    int m_rows = 0, n_cols = 0;
    int nnz = 0;
    int *d_col_offsets = nullptr;
    int *d_row_indices = nullptr;
    T *d_vals = nullptr;

    size_t mem_size_col_offsets() const {
        return n_cols*sizeof(int);
    }

    size_t mem_size_row_indices() const {
        return nnz*sizeof(int);
    }

    size_t mem_size_vals() const {
        return nnz*sizeof(T);
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

    int binary_search_for_target_index(int target, int *arr, int start, int end) const {

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
                return binary_search_for_target_index(target, arr, cand_ind+1, end);
            } else {
                return binary_search_for_target_index(target, arr, start, cand_ind);
            }
        }

    }

    ImmutableMatrixSparse<__half> to_half() const;
    ImmutableMatrixSparse<float> to_float() const;
    ImmutableMatrixSparse<double> to_double() const;

public:

    class Block; class Col; // Forward declaration of nested classes

    // *** Basic Constructors ***
    ImmutableMatrixSparse(
        const cuHandleBundle &arg_cu_handles,
        int arg_m,
        int arg_n
    ):
        cu_handles(arg_cu_handles),
        m_rows(arg_m),
        n_cols(arg_n),
        nnz(0)
    {

        allocate_d_mem();

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        for (int i=0; i<n_cols; ++i) { h_col_offsets[i] = 0; }
        check_cuda_error(cudaMemcpy(
            d_col_offsets, h_col_offsets, mem_size_col_offsets(), cudaMemcpyHostToDevice
        ));
        free(h_col_offsets);

    }

    ImmutableMatrixSparse(const cuHandleBundle &arg_cu_handles):
        ImmutableMatrixSparse(arg_cu_handles, 0, 0)
    {}

    ImmutableMatrixSparse(
        const cuHandleBundle &arg_cu_handles,
        std::initializer_list<std::initializer_list<T>> li
    ):
        cu_handles(arg_cu_handles),
        m_rows(li.size()),
        n_cols((li.size() == 0) ? 0 : std::cbegin(li)->size())
    {

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        std::vector<int> vec_row_indices;
        std::vector<T> vec_values;

        // Capture iterators for each row
        std::vector<typename std::initializer_list<T>::const_iterator> row_iters;
        for (
            typename std::initializer_list<std::initializer_list<T>>::const_iterator curr_row = std::cbegin(li);
            curr_row != std::cend(li);
            ++curr_row
        ) {
            // Check all row iterators have consistent size before adding
            if (curr_row->size() == n_cols) {
                row_iters.push_back(std::cbegin(*curr_row));
            } else {
                throw std::runtime_error(
                    "ImmutableMatrixSparse: Non-consistent row size encounted in initializer list"
                );
            }
        }

        // Roll through iterators repeatedly to access non-zero elements column by column calculating
        // offsets based on number of nnz values encountered
        int next_col_offset = 0;
        for (int j=0; j<n_cols; ++j) {

            h_col_offsets[j] = next_col_offset;
            for (int i=0; i<m_rows; ++i) {

                T val = *(row_iters[i]);
                if (val != static_cast<T>(0.)) {
                    vec_row_indices.push_back(i);
                    vec_values.push_back(val);
                    ++next_col_offset;
                }
                ++row_iters[i];

            }

        }
        nnz = vec_values.size();

        // Set remaining host vectors to values to load, and load column offsets, row indices
        // and values into gpu memory
        allocate_d_mem();
        int *h_row_indices = static_cast<int *>(malloc(mem_size_row_indices()));
        T *h_vals = static_cast<T *>(malloc(mem_size_vals()));
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

    // *** Copy Constructor ***
    ImmutableMatrixSparse(const ImmutableMatrixSparse &other) {
        *this = other;
    }

    // *** Destructor *** 
    ~ImmutableMatrixSparse() {
        check_cuda_error(cudaFree(d_col_offsets));
        check_cuda_error(cudaFree(d_row_indices));
        check_cuda_error(cudaFree(d_vals));
    }

    // *** Copy Assignment ***
    ImmutableMatrixSparse & operator=(const ImmutableMatrixSparse &other) {

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

    // *** Dynamic Memory *** (assumes outer code handles dynamic memory properly)
    ImmutableMatrixSparse(
        const cuHandleBundle &source_cu_handles,
        int *h_col_offsets,
        int *h_row_indices,
        T *h_vals,
        int source_m_rows,
        int source_n_cols,
        int source_nnz
    ):
        cu_handles(source_cu_handles),
        m_rows(source_m_rows),
        n_cols(source_n_cols),
        nnz(source_nnz)
    {

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
        int *h_col_offsets, int *h_row_indices, T *h_vals,
        int target_m_rows, int target_n_cols, int target_nnz
    ) const {

        if (target_m_rows != m_rows) {
            throw std::runtime_error("ImmutableMatrixSparse: invalid target_m_rows dim for copy_data_to_ptr");
        }
        if (target_n_cols != n_cols) {
            throw std::runtime_error("ImmutableMatrixSparse: invalid target_n_cols dim for copy_data_to_ptr");
        }
        if (target_nnz != nnz) {
            throw std::runtime_error("ImmutableMatrixSparse: invalid target_nnz dim for copy_data_to_ptr");
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

    // *** Element Access ***
    const Scalar<T> get_elem(int row, int col) const {

        if ((row < 0) || (row >= m_rows)) {
            throw std::runtime_error("ImmutableMatrixSparse: invalid row access in get_elem");
        }
        if ((col < 0) || (col >= n_cols)) {
            throw std::runtime_error("ImmutableMatrixSparse: invalid col access in get_elem");
        }

        // Get column offset and column size
        int col_offset_L;
        int col_offset_R;
        if (col != n_cols-1) {
            check_cuda_error(cudaMemcpy(
                &col_offset_L, d_col_offsets+col, sizeof(int), cudaMemcpyDeviceToHost
            ));
            check_cuda_error(cudaMemcpy(
                &col_offset_R, d_col_offsets+col+1, sizeof(int), cudaMemcpyDeviceToHost
            ));
        } else {
            check_cuda_error(cudaMemcpy(
                &col_offset_L, d_col_offsets+col, sizeof(int), cudaMemcpyDeviceToHost
            ));
            col_offset_R = nnz;
        }
        size_t col_nnz_size = col_offset_R-col_offset_L;

        // Find if row index is non-zero and find location in val array
        int *h_row_indices_for_col = static_cast<int *>(malloc(col_nnz_size*sizeof(int)));
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
            Scalar<T> elem;
            check_cuda_error(cudaMemcpy(
                elem.d_scalar, d_vals+col_offset_L+row_ind_offset, sizeof(T), cudaMemcpyDeviceToDevice
            ));
            return elem;
        } else {
            return Scalar<T>(static_cast<T>(0.));
        }

    }

//     Col col(int _col) { return Parent::col(_col); }
//     Block block(int row, int col, int m, int n) { return Parent::block(row, col, m, n); }

    // *** Properties ***
    cuHandleBundle get_cu_handles() const { return cu_handles; }
    int rows() const { return m_rows; }
    int cols() const { return n_cols; }
    int non_zeros() const { return nnz; }

    std::string get_matrix_string() const {

        int *h_col_offsets = static_cast<int *>(malloc(mem_size_col_offsets()));
        int *h_row_indices = static_cast<int *>(malloc(mem_size_row_indices()));
        T *h_vals = static_cast<T *>(malloc(mem_size_vals()));

        copy_data_to_ptr(
            h_col_offsets, h_row_indices, h_vals,
            m_rows, n_cols, nnz
        );

        std::string col_offsets_str = "[";
        if (n_cols > 0) {
            for (int i=0; i<n_cols-1; ++i) { col_offsets_str += std::format("{}, ", h_col_offsets[i]); }
            col_offsets_str += std::format("{}", h_col_offsets[n_cols-1]);
        }
        col_offsets_str += "]";

        std::string row_indices_str = "[";
        std::string val_str = "[";
        if (nnz > 0) {
            for (int i=0; i<nnz-1; ++i) {
                row_indices_str += std::format("{}, ", h_row_indices[i]);
                val_str += std::format("{:.6g}, ", static_cast<double>(h_vals[i]));
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
            "Rows: {} | Cols: {} | Non-zeroes: {} | Fill ratio: {:.3g} | Max magnitude: {:.3g}",
            m_rows,
            n_cols,
            nnz,
            static_cast<double>(nnz)/static_cast<double>(m_rows*n_cols),
            static_cast<double>(get_max_mag_elem().get_scalar())
        );
    }

    // *** Static Creation ***
    static ImmutableMatrixSparse<T> Zero(
        const cuHandleBundle &arg_cu_handles, int arg_m_rows, int arg_n_cols
    ) {
        return ImmutableMatrixSparse<T>(arg_cu_handles, arg_m_rows, arg_n_cols);
    }

    static ImmutableMatrixSparse<T> Ones(
        const cuHandleBundle &arg_cu_handles, int arg_m_rows, int arg_n_cols
    ) {

        int *h_col_offsets = static_cast<int *>(malloc(arg_n_cols*sizeof(int)));
        int *h_row_indices = static_cast<int *>(malloc(arg_m_rows*arg_n_cols*sizeof(int)));
        T *h_vals = static_cast<T *>(malloc(arg_m_rows*arg_n_cols*sizeof(T)));

        for (int i=0; i<arg_n_cols; ++i) { h_col_offsets[i] = i*arg_m_rows; }
        for (int i=0; i<arg_m_rows*arg_n_cols; ++i) { h_row_indices[i] = i%arg_m_rows; }
        for (int i=0; i<arg_m_rows*arg_n_cols; ++i) { h_vals[i] = static_cast<T>(1.); }

        ImmutableMatrixSparse<T> ret_mat(
            arg_cu_handles,
            h_col_offsets, h_row_indices, h_vals,
            arg_m_rows, arg_n_cols, arg_m_rows*arg_n_cols
        );

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return ret_mat;

    }

    static ImmutableMatrixSparse<T> Identity(
        const cuHandleBundle &arg_cu_handles, int arg_m_rows, int arg_n_cols
    ) {

        int smaller_dim = std::min(arg_m_rows, arg_n_cols);

        int *h_col_offsets = static_cast<int *>(malloc(arg_n_cols*sizeof(int)));
        int *h_row_indices = static_cast<int *>(malloc(smaller_dim*sizeof(int)));
        T *h_vals = static_cast<T *>(malloc(smaller_dim*sizeof(T)));

        if (arg_n_cols == smaller_dim) {
            for (int i=0; i<arg_n_cols; ++i) { h_col_offsets[i] = i; }
        } else {
            for (int i=0; i<smaller_dim; ++i) { h_col_offsets[i] = i; }
            for (int i=smaller_dim; i<arg_n_cols; ++i) { h_col_offsets[i] = smaller_dim; }
        }

        for (int i=0; i<smaller_dim; ++i) { h_row_indices[i] = i; }
    
        for (int i=0; i<smaller_dim; ++i) { h_vals[i] = static_cast<T>(1.); }

        ImmutableMatrixSparse<T> ret_mat(
            arg_cu_handles,
            h_col_offsets, h_row_indices, h_vals,
            arg_m_rows, arg_n_cols, smaller_dim
        );

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        return ret_mat;

    }

    static ImmutableMatrixSparse<T> Random(
        const cuHandleBundle &arg_cu_handles, int arg_m_rows, int arg_n_cols, double fill_prob
    ) {

        if (!((0. <= fill_prob) && (fill_prob <= 1.))) {
            throw std::runtime_error("ImmutableMatrixSparse: Random invalid fill_prob");
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> val_dist(-1., 1.);
        std::uniform_real_distribution<double> fill_prob_dist(0., 1.);

        int *h_col_offsets = static_cast<int *>(malloc(arg_n_cols*sizeof(int)));
        std::vector<int> h_vec_row_indices;
        std::vector<T> h_vec_vals;

        int curr_nnz = 0;
        for (int j=0; j<arg_n_cols; ++j) {
            h_col_offsets[j] = curr_nnz;
            for (int i=0; i<arg_m_rows; ++i) {
                if ((fill_prob != 0.) && (fill_prob_dist(gen) <= fill_prob)) {
                    T val = val_dist(gen);
                    if (val != static_cast<T>(0.)) {
                        h_vec_row_indices.push_back(i);
                        h_vec_vals.push_back(val);
                        ++curr_nnz;
                    }
                }
            }
        }

        ImmutableMatrixSparse<T> created_mat(arg_cu_handles);
        if (curr_nnz != 0) {
            created_mat = ImmutableMatrixSparse<T>(
                arg_cu_handles,
                h_col_offsets, &h_vec_row_indices[0], &h_vec_vals[0],
                arg_m_rows, arg_n_cols, curr_nnz
            );
        } else {
            created_mat = ImmutableMatrixSparse<T>(arg_cu_handles, arg_m_rows, arg_n_cols);
        }

        free(h_col_offsets);

        return created_mat;

    }

//     // *** Resizing ***
//     void reduce() { Parent::prune(static_cast<T>(0)); }

//     // *** Explicit Cast ***
//     template <typename Cast_T>
//     MatrixSparse<Cast_T> cast() const {
//         return typename Eigen::SparseMatrix<Cast_T>::SparseMatrix(
//             Parent::template cast<Cast_T>()
//         );
//     }

    // *** Arithmetic and Compound Operations ***

    Scalar<T> get_max_mag_elem() const {

        T *h_vals = static_cast<T *>(malloc(mem_size_vals()));
        if (nnz > 0) {
            check_cuda_error(cudaMemcpy(
                h_vals,
                d_vals,
                mem_size_vals(),
                cudaMemcpyDeviceToHost
            ));
        }

        T max_mag = static_cast<T>(0);
        for (int i=0; i<nnz; ++i) {
            T temp = static_cast<T>(std::abs(static_cast<double>(h_vals[i])));
            if (max_mag < temp) {
                max_mag = temp;
            }
        }

        free(h_vals);

        return Scalar<T>(max_mag);

    }

//     MatrixSparse<T> transpose() const {
//         return typename Parent::SparseMatrix(Parent::transpose());
//     }
//     MatrixSparse<T> operator*(const T &scalar) const {
//         return typename Parent::SparseMatrix(Parent::operator*(scalar));
//     }
//     MatrixSparse<T> operator/(const T &scalar) const {
//         return typename Parent::SparseMatrix(Parent::operator/(scalar));
//     }
//     MatrixVector<T> operator*(const MatrixVector<T> &vec) const {
//         return typename Matrix<T, Dynamic, 1>::Matrix(Parent::operator*(vec.base()));
//     }
//     T norm() const { return Parent::norm(); } // Needed for testing
//     MatrixSparse<T> operator+(const MatrixSparse<T> &mat) const { // Needed for testing
//         return typename Parent::SparseMatrix(Parent::operator+(mat));
//     }
//     MatrixSparse<T> operator-(const MatrixSparse<T> &mat) const { // Needed for testing
//         return typename Parent::SparseMatrix(Parent::operator-(mat));
//     }
//     MatrixSparse<T> operator*(const MatrixSparse<T> &mat) const { // Needed for testing
//         return typename Parent::SparseMatrix(Parent::operator*(mat));
//     }

//     // Forward iterator over sparse inner columns skipping zeroes
//     class InnerIterator: public Parent::InnerIterator
//     {
//     public:
        
//         InnerIterator(const MatrixSparse<T> &mat, int start):
//             Parent::InnerIterator(mat, start)
//         {}

//         int col() { return Parent::InnerIterator::col(); }
//         int row() { return Parent::InnerIterator::row(); }
//         T value() { return Parent::InnerIterator::value(); }
//         typename Parent::InnerIterator &operator++() {
//             return Parent::InnerIterator::operator++();
//         }
//         operator bool() const { return Parent::InnerIterator::operator bool(); }

//     };

//     // Reverse iterator over sparse inner columns skipping zeroes
//     class ReverseInnerIterator: public Parent::ReverseInnerIterator
//     {
//     public:
        
//         ReverseInnerIterator(const MatrixSparse<T> &mat, int start):
//             Parent::ReverseInnerIterator(mat, start)
//         {}

//         int col() { return Parent::ReverseInnerIterator::col(); }
//         int row() { return Parent::ReverseInnerIterator::row(); }
//         T value() { return Parent::ReverseInnerIterator::value(); }
//         typename Parent::ReverseInnerIterator &operator--() {
//             return Parent::ReverseInnerIterator::operator--();
//         }
//         operator bool() const { return Parent::ReverseInnerIterator::operator bool(); }

//     };

//     // Nested class representing sparse matrix column
//     // Requires: assignment from/cast to MatrixVector<T>
//     class Col: private Eigen::Block<Parent, Eigen::Dynamic, 1, true>
//     {
//     private:

//         using ColParent = Eigen::Block<Parent, Eigen::Dynamic, 1, true>;
//         friend MatrixVector<T>;
//         friend MatrixSparse<T>;
//         const ColParent &base() const { return *this; }
//         Col(const ColParent &other): ColParent(other) {}

//     public:

//         Col operator=(const MatrixVector<T> vec) { return ColParent::operator=(vec.base().sparseView()); }

//     };

//     // Nested class representing sparse matrix block
//     // Requires: cast to MatrixDense<T>
//     class Block: private Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>
//     {
//     private:

//         using BlockParent = Eigen::Block<Parent, Eigen::Dynamic, Eigen::Dynamic>;
//         friend MatrixDense<T>;
//         friend MatrixSparse<T>;
//         const BlockParent &base() const { return *this; }
//         Block(const BlockParent &other): BlockParent(other) {}

//     };
    
// };

// #endif

// template <typename T>
// MatrixVector<T> back_substitution(MatrixSparse<T> const &UT, MatrixVector<T> const &rhs) {

//     // Check squareness and compatibility
//     if (UT.rows() != UT.cols()) { throw std::runtime_error("Non square matrix in back substitution"); }
//     if (UT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

//     // Assume UT is upper triangular, iterate backwards through columns through non-zero entries
//     // for backward substitution
//     MatrixVector<T> x = rhs;
//     for (int col=UT.cols()-1; col>=0; --col) {

//         typename MatrixSparse<T>::ReverseInnerIterator it(UT, col);
        
//         // Skip entries until reaching diagonal guarding against extra non-zeroes
//         for (; it && (it.row() != it.col()); --it) { ; }
//         if (it.row() != it.col()) { throw std::runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }

//         x(col) /= it.value();
//         --it;
//         for (; it; --it) { x(it.row()) -= it.value()*x(col); }

//     }

//     return x;

// }
// template <typename T>
// MatrixVector<T> frwd_substitution(MatrixSparse<T> const &LT, MatrixVector<T> const &rhs) {

//     // Check squareness and compatibility
//     if (LT.rows() != LT.cols()) { throw std::runtime_error("Non square matrix in back substitution"); }
//     if (LT.rows() != rhs.rows()) { throw std::runtime_error("Incompatible matrix and rhs"); }

//     // Assume LT is lower triangular, iterate forwards through columns through non-zero entries
//     // for forward substitution
//     MatrixVector<T> x = rhs;
//     for (int col=0; col<LT.cols(); ++col) {

//         typename MatrixSparse<T>::InnerIterator it(LT, col);
        
//         // Skip entries until reaching diagonal guarding against extra non-zeroes
//         for (; it && (it.row() != it.col()); ++it) { ; }
//         if (it.row() != it.col()) { throw std::runtime_error ("Diagonal in MatrixSparse triangular solve could not be reached"); }
        
//         x(col) /= it.value();
//         ++it;
//         for (; it; ++it) { x(it.row()) -= it.value()*x(col); }

//     }

//     return x;

};

#endif
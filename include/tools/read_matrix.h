#ifndef MATRIX_READER_H
#define MATRIX_READER_H

#include <memory>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "types/types.h"
#include "tools/cuHandleBundle.h"
#include "tools/DenseConverter.h"

namespace fs = std::filesystem;

template <template<typename> typename M, typename T>
M<T> read_matrixCSV(const cuHandleBundle &cu_handles, fs::path const &path) {

    // Open given file
    std::ifstream file_in;
    file_in.open(path);

    // Ensure read success
    if (!file_in.is_open()) {
        throw std::runtime_error("read_matrixCSV: failed to read: " + path.string());
    }
    
    int m_rows = 0;
    int n_cols = 0;
    std::string line_string;
    std::stringstream line_stream;
    bool is_first_line = true;
    std::string temp_str;

    // Scan file getting m_rows and n_cols and ensuring rectangular nature
    while (std::getline(file_in, line_string)) {
        
        ++m_rows;
        line_stream.clear();
        line_stream << line_string;

        if (is_first_line) {
            // Iterate over first row to get number of columns
            while (std::getline(line_stream, temp_str, ',')) { ++n_cols; }
            is_first_line = false;
        } else {
            // Ensure subsequent rows meet col count of first column
            int col_count = 0;
            while (std::getline(line_stream, temp_str, ',')) { ++col_count; }
            if (col_count != n_cols) {
                throw std::runtime_error(
                    "read_matrixCSV: error in: " + path.string() + "\n" +
                    "row " + std::to_string(m_rows) +
                    " does not meet column size of " + std::to_string(n_cols)
                );
            }
        }
    
    }

    // Read entries into matrix
    file_in.clear();
    file_in.seekg(0, std::ios_base::beg);

    T *h_mat = static_cast<T *>(malloc(m_rows*n_cols*sizeof(T)));

    T temp_number;
    int row = 0;
    while (std::getline(file_in, line_string)) {

        line_stream.clear();
        line_stream << line_string;

        int col = 0;
        while (std::getline(line_stream, temp_str, ',')) {
            try {
                temp_number = static_cast<T>(stod(temp_str));
            } catch (std::invalid_argument e) {
                free(h_mat);
                throw std::runtime_error(
                    "read_matrixCSV: error in: " + path.string() + "\n" +
                    "Invalid argument in file, failed to convert to numeric"
                );
            }
            h_mat[row+col*m_rows] = temp_number;
            col++;
        }
        row++;

    }

    MatrixDense<T> mat(cu_handles, h_mat, m_rows, n_cols);

    free(h_mat);

    DenseConverter<M, T> converter;
    return converter.convert_matrix(mat);

}

template <typename T>
NoFillMatrixSparse<T> read_matrixMTX(const cuHandleBundle &cu_handles, fs::path const &path) {

    return NoFillMatrixSparse<T>(cuHandleBundle());

    // // Open given file
    // std::ifstream file_in;
    // file_in.open(path);

    // // Ensure path is a matrix market mtx
    // if (path.extension() == fs::path(".mtx")) {
    //     throw std::runtime_error("read_matrixMTX: non .mtx file given: " + path.string());
    // }

    // // Ensure read success
    // if (!file_in.is_open()) {
    //     throw std::runtime_error("read_matrixMTX: failed to read: " + path.string());
    // }

    // std::string line_str;
    // std::string line_str_portion;
    // std::stringstream line_in;

    // // Read first line to affirm format and get symmetry
    // std::string first_line_error("read_matrixMTX: first line incorrect to expected format");
    // std::getline(file_in, line_str);
    // std::getline(line_in, line_str_portion, " ");
    // if (line_str_portion != "%%MatrixMarket") { throw std::runtime_error(first_line_error); }
    // std::getline(line_in, line_str_portion, " ");
    // if (line_str_portion != "matrix") { throw std::runtime_error(first_line_error); }
    // std::getline(line_in, line_str_portion, " ");
    // if (line_str_portion != "coordinate") { throw std::runtime_error(first_line_error); }
    // std::getline(line_in, line_str_portion, " ");
    // if (line_str_portion != "real") { throw std::runtime_error(first_line_error); }

    // // Read symmetry of mtx
    // bool symmetric;
    // std::getline(line_in, line_str_portion);
    // if (line_str_portion == "general") {
    //     symmetric == false;
    // } else if (line_str_portion == "symmetric") {
    //     symmetric == true;
    // } else {
    //     throw std::runtime_error(first_line_error);
    // }

    // // Skip word description %
    // std::getline(file_in, line_string);
    // while (line_string[0] == "%") { std::getline(file_in, line_string); }

    // // Read matrix dimensions and initiate data storage
    // int m_rows;
    // int n_cols;
    // int entries;
    // std::string mtx_dim_error("read_matrixMTX: matrix dimensions incorrect to expected format");
    // try {
    //     std::getline(line_in, line_str_portion, " ");
    //     m_rows = stoi(line_str_portion);
    //     std::getline(line_in, line_str_portion, " ");
    //     n_cols = stoi(line_str_portion);
    //     std::getline(line_in, line_str_portion);
    //     entries = stoi(line_str_portion);
    // } catch (std::invalid_argument e) {
    //     throw std::runtime_error(mtx_dim_error);
    // }

    // int total_nnz = 0;
    // std::vector<std::vector<int>> vec_row_indices(n);
    // std::vector<std::vector<T>> vec_vals(n);

    // // Load data
    // for (int entry=0; entry<entries; ++entry) {

    //     try {

    //         std::getline(line_in, line_str_portion, " ");
    //         int i = stoi(line_str_portion)-1; // 1-indexing correction
    //         std::getline(line_in, line_str_portion, " ");
    //         int j = stoi(line_str_portion)-1; // 1-indexing correction
    //         std::getline(line_in, line_str_portion);
    //         T val = static_cast<T>(stod(line_str_portion));

    //         if (val != static_cast<T>(0.)) {
    //             vec_row_indices[j].push_back(i);
    //             vec_vals[j].push_back(val);
    //             ++total_nnz;
    //             if ((symmetric) && (i > j)) {
    //                 vec_row_indices[i].push_back(j);
    //                 vec_vals[i].push_back(val);
    //                 ++total_nnz;
    //             }
    //         }

    //     } catch (std::invalid_argument e) {

    //         throw std::runtime_error(
    //             "read_matrixMTX: Invalid argument in file, failed to convert to numeric"
    //         );

    //     }

    // }

    // int curr_nnz = 0;
    // int *h_col_offsets = static_cast<int *>(malloc((n_cols+1)*sizeof(int)));
    // int *h_row_indices = static_cast<int *>(malloc(total_nnz*sizeof(int)));
    // T *h_vals = static_cast<T *>(malloc(total_nnz*sizeof(T)));

    // for (int j=0; j<n_cols; ++j) {
    //     h_col_offsets[j] = curr_nnz;
    //     for (int i=0; i<vec_row_indices[j].size(); ++i) {
    //         h_row_indices[curr_nnz+i] = vec_row_indices[j][i];
    //         h_vals[curr_nnz+i] = vec_vals[j][i];
    //     }
    //     curr_nnz += vec_row_indices[j].size();
    // }
    // assert(curr_nnz == total_nnz, "read_matrixMTX: mismatch in matrix filling of nnz count");
    // h_col_offsets[n_cols] = curr_nnz;

    // NoFillMatrixSparse<T> mat(
    //     m_rows, n_cols, total_nnz,
    //     h_col_offsets, h_row_indices, h_vals
    // );

    // free(h_col_offsets);
    // free(h_row_indices);
    // free(h_vals);

    // return mat;

}

#endif
#ifndef MATRIX_READER_H
#define MATRIX_READER_H

#include "tools/cuHandleBundle.h"
#include "types/types.h"
#include "tools/DenseConverter.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

namespace fs = std::filesystem;

template <template <typename> typename TMatrix, typename TPrecision>
TMatrix<TPrecision> read_matrixCSV(
    const cuHandleBundle &cu_handles, fs::path const &path
) {

    // Open given file
    std::ifstream file_in;
    file_in.open(path);

    // Ensure read success
    if (!file_in.is_open()) {
        throw std::runtime_error(
            "read_matrixCSV: failed to read: " + path.string()
        );
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
                    "read_matrixCSV: error in: " + path.string() + "\n"
                    "row " + std::to_string(m_rows) +
                    " does not meet column size of " + std::to_string(n_cols)
                );
            }
        }
    
    }

    // Read entries into matrix
    file_in.clear();
    file_in.seekg(0, std::ios_base::beg);

    TPrecision *h_mat = static_cast<TPrecision *>(
        malloc(m_rows*n_cols*sizeof(TPrecision))
    );

    TPrecision temp_number;
    int row = 0;
    while (std::getline(file_in, line_string)) {

        line_stream.clear();
        line_stream << line_string;

        int col = 0;
        while (std::getline(line_stream, temp_str, ',')) {
            try {
                temp_number = static_cast<TPrecision>(stod(temp_str));
            } catch (std::invalid_argument e) {
                free(h_mat);
                throw std::runtime_error(
                    "read_matrixCSV: error in: " + path.string() + "\n"
                    "Invalid argument in file, failed to convert to numeric"
                );
            }
            h_mat[row+col*m_rows] = temp_number;
            col++;
        }
        row++;

    }

    MatrixDense<TPrecision> mat(cu_handles, h_mat, m_rows, n_cols);

    free(h_mat);

    DenseConverter<TMatrix, TPrecision> converter;

    return converter.convert_matrix(mat);

}

template <template <typename> typename TMatrix, typename TPrecision>
TMatrix<TPrecision> read_matrixMTX(
    const cuHandleBundle &cu_handles, fs::path const &path
) {

    // Open given file
    std::ifstream file_in;
    file_in.open(path);

    // Ensure path is a matrix market mtx
    if (path.extension() != fs::path(".mtx")) {
        throw std::runtime_error(
            "read_matrixMTX: non .mtx file given: " + path.string()
        );
    }

    // Ensure read success
    if (!file_in.is_open()) {
        throw std::runtime_error(
            "read_matrixMTX: failed to read: " + path.string()
        );
    }

    std::string line_str;
    std::string line_str_portion;
    std::stringstream line_in;

    auto no_whitespace = [](std::string &str) -> bool {
        for (auto iter = str.cbegin(); iter != str.cend(); ++iter) {
            if (std::isspace(*iter)) { return false; }
        }
        return true;
    };

    // Read first line to affirm format and get symmetry
    std::string first_line_error(
        "read_matrixMTX: first line incorrect to expected format"
    );
    std::getline(file_in, line_str);
    line_in.clear();
    line_in << line_str;
    if (
        !(std::getline(line_in, line_str_portion, ' ')) ||
        (line_str_portion != "%%MatrixMarket")
    ) {
        throw std::runtime_error(first_line_error+" %%MatrixMarket");
    }
    if (
        !(std::getline(line_in, line_str_portion, ' ')) ||
        (line_str_portion != "matrix")
    ) {
        throw std::runtime_error(first_line_error+" matrix");
    }
    if (
        !(std::getline(line_in, line_str_portion, ' ')) ||
        (line_str_portion != "coordinate")
    ) {
        throw std::runtime_error(first_line_error+" coordinate");
    }
    if (
        !(std::getline(line_in, line_str_portion, ' ')) ||
        (line_str_portion != "real")
    ) {
        throw std::runtime_error(first_line_error+" real");
    }

    // Read symmetry of mtx
    bool symmetric;
    std::getline(line_in, line_str_portion);
    if (line_str_portion == "general") {
        symmetric = false;
    } else if (line_str_portion == "symmetric") {
        symmetric = true;
    } else {
        throw std::runtime_error(first_line_error);
    }

    // Skip word description %
    std::getline(file_in, line_str);
    while (line_str[0] == '%') { std::getline(file_in, line_str); }
    line_in.clear();
    line_in << line_str;

    // Read matrix dimensions and initiate data storage
    int m_rows;
    int n_cols;
    int entries;
    std::string mtx_dim_error(
        "read_matrixMTX: matrix dimensions incorrect to expected format"
    );
    try {
        if (
            !std::getline(line_in, line_str_portion, ' ') ||
            !no_whitespace(line_str_portion)
        ) {
            throw std::invalid_argument("");
        }
        m_rows = stoi(line_str_portion);
        if (
            !std::getline(line_in, line_str_portion, ' ') ||
            !no_whitespace(line_str_portion)
        ) {
            throw std::invalid_argument("");
        }
        n_cols = stoi(line_str_portion);
        if (
            !std::getline(line_in, line_str_portion) ||
            !no_whitespace(line_str_portion)
        ) {
            throw std::invalid_argument("");
        }
        entries = stoi(line_str_portion);
        if ((m_rows < 0) || (n_cols < 0) || (entries < 0)) {
            throw std::invalid_argument("");
        }
    } catch (std::invalid_argument e) {
        throw std::runtime_error(mtx_dim_error);
    }

    int total_nnz = 0;
    std::vector<std::vector<int>> vec_row_indices(n_cols);
    std::vector<std::vector<TPrecision>> vec_vals(n_cols);

    // Load data and check validity
    int last_i = -1;
    int last_j = -1;
    for (int entry=0; entry<entries; ++entry) {

        try {

            std::getline(file_in, line_str);
            line_in.clear();
            line_in << line_str;

            if (
                !std::getline(line_in, line_str_portion, ' ') ||
                !no_whitespace(line_str_portion)
            ) {
                throw std::invalid_argument("");
            }
            int i = stoi(line_str_portion)-1; // 1-indexing correction
            if (
                !std::getline(line_in, line_str_portion, ' ') ||
                !no_whitespace(line_str_portion)
            ) {
                throw std::invalid_argument("");
            }
            int j = stoi(line_str_portion)-1; // 1-indexing correction
            if (
                !std::getline(line_in, line_str_portion) ||
                !no_whitespace(line_str_portion)
            ) {
                throw std::invalid_argument("");
            }
            TPrecision val = static_cast<TPrecision>(stod(line_str_portion));

            // Check validity of entry values
            if (
                (j == last_j) &&
                ((i <= last_i) || (i < 0) || (i >= m_rows))
            ) {
                throw std::runtime_error(
                    "read_matrixMTX: invalid row order encountered"
                );
            } else if ((j < last_j) || (j < 0) || (j >= n_cols)) {
                throw std::runtime_error(
                    "read_matrixMTX: invalid column encountered"
                );
            } else if ((i < 0) || (i >= m_rows)) {
                throw std::runtime_error(
                    "read_matrixMTX: invalid row encountered"
                );
            } else if (symmetric && (j > i)) {
                throw std::runtime_error(
                    "read_matrixMTX: above diagonal entry in symmetric"
                );
            }
            last_i = i;
            last_j = j;

            if (val != static_cast<TPrecision>(0.)) {

                vec_row_indices[j].push_back(i);
                vec_vals[j].push_back(val);
                ++total_nnz;

                if (symmetric && (i > j)) {
                    vec_row_indices[i].push_back(j);
                    vec_vals[i].push_back(val);
                    ++total_nnz;
                }

            }

        } catch (std::invalid_argument e) {
            throw std::runtime_error(
                std::format(
                    "read_matrixMTX: Invalid entry in file entry: {}", 
                    entry
                )
            );
        }

    }

    int curr_nnz = 0;
    int *h_col_offsets = static_cast<int *>(malloc((n_cols+1)*sizeof(int)));
    int *h_row_indices = static_cast<int *>(malloc(total_nnz*sizeof(int)));
    TPrecision *h_vals = static_cast<TPrecision *>(
        malloc(total_nnz*sizeof(TPrecision))
    );

    for (int j=0; j<n_cols; ++j) {
        h_col_offsets[j] = curr_nnz;
        for (int i=0; i<vec_row_indices[j].size(); ++i) {
            h_row_indices[curr_nnz+i] = vec_row_indices[j][i];
            h_vals[curr_nnz+i] = vec_vals[j][i];
        }
        curr_nnz += vec_row_indices[j].size();
    }
    assert(curr_nnz == total_nnz);
    h_col_offsets[n_cols] = curr_nnz;

    NoFillMatrixSparse<TPrecision> mat(
        cu_handles,
        h_col_offsets, h_row_indices, h_vals,
        m_rows, n_cols, total_nnz
    );

    free(h_col_offsets);
    free(h_row_indices);
    free(h_vals);

    return mat;

}

#endif
#ifndef MATRIXREADER_H
#define MATRIXREADER_H

#include "types/types.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

template <template<typename> typename M, typename T>
M<T> read_matrixCSV(const cublasHandle_t &handle, fs::path const &path) {

    assert_valid_type_or_vec<M>();

    // Open given file
    std::ifstream file_in;
    file_in.open(path);

    // Ensure read success
    if (!file_in.is_open()) {
        throw std::runtime_error("read_matrixCSV: failed to read: " + path.string());
    }
    
    int n_rows = 0;
    int n_cols = 0;
    std::string line_string;
    std::stringstream line_stream;
    bool is_first_line = true;
    std::string temp_str;

    // Scan file getting n_rows and n_cols and ensuring rectangular nature
    while (std::getline(file_in, line_string)) {
        
        ++n_rows;
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
                    "row " + std::to_string(n_rows) +
                    " does not meet column size of " + std::to_string(n_cols)
                );
            }
        }
    
    }

    // Read entries into matrix
    file_in.clear();
    file_in.seekg(0, std::ios_base::beg);
    M<T> mat(M<T>::Zero(handle, n_rows, n_cols));
    T temp_number;
    int row = 0;

    while (std::getline(file_in, line_string)) {

        line_stream.clear();
        line_stream << line_string;

        int col = 0;
        while (std::getline(line_stream, temp_str, ',')) {
            try { temp_number = static_cast<T>(stod(temp_str)); }
            catch (std::invalid_argument e) {
                throw std::runtime_error(
                    "read_matrixCSV: error in: " + path.string() + "\n" +
                    "Invalid argument in file, failed to convert to numeric"
                );
            }
            if (temp_number != static_cast<T>(0)) { mat.set_elem(row, col, temp_number); }
            col++;
        }

        row++;

    }

    mat.reduce();
    return mat;

}


#endif
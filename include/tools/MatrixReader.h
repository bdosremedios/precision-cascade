#ifndef MATRIXREADER_H
#define MATRIXREADER_H

#include "types/types.h"

#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;
using std::string, std::to_string;
using std::ifstream, std::stringstream, std::getline;
using std::runtime_error;
using std::cout, std::endl;

template <template<typename> typename M, typename T>
M<T> read_matrixCSV(fs::path const &path) {

    assert_valid_type_or_vec<M>();

    // Open given file
    ifstream file_in;
    file_in.open(path);

    // Ensure success
    if (!file_in.is_open()) { throw runtime_error("Failed to read: " + path.string()); }
    
    int n_rows = 0;
    int n_cols = 0;
    string line_string;
    stringstream line_stream;
    bool is_first_line = true;
    string temp_str;

    // Scan file getting n_rows and n_cols and ensuring rectangular nature
    while (getline(file_in, line_string)) {
        
        ++n_rows;
        line_stream.clear();
        line_stream << line_string;

        if (is_first_line) {
            // Iterate over first row to get number of columns
            while (getline(line_stream, temp_str, ',')) { ++n_cols; }
            is_first_line = false;
        } else {
            // Ensure subsequent rows meet col count of first column
            int col_count = 0;
            while (getline(line_stream, temp_str, ',')) { ++col_count; }
            if (col_count != n_cols) {
                throw runtime_error(
                    "Error in: " + path.string() + "\n" +
                    "Row " + to_string(n_rows) + " does not meet column size of " + to_string(n_cols)
                );
            }
        }
    
    }

    // Read entries into matrix
    file_in.clear();
    file_in.seekg(0, std::ios_base::beg);
    M<T> mat(M<T>::Zero(n_rows, n_cols));
    T temp_number;
    int row = 0;

    while (getline(file_in, line_string)) {

        line_stream.clear();
        line_stream << line_string;

        int col = 0;
        while (getline(line_stream, temp_str, ',')) {
            try { temp_number = static_cast<T>(stod(temp_str)); }
            catch (std::invalid_argument e) {
                throw runtime_error(
                    "Error in: " + path.string() + "\n" +
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
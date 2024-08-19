#include "tools/read_matrix.h"

std::ifstream cascade::helper::open_ifstream(
    fs::path path, fs::path correct_extension
) {

    std::ifstream file_in;
    file_in.open(path);

    if (path.extension() != correct_extension) {
        throw std::runtime_error(
            "open_ifstream: incorrect file given: " +
            correct_extension.string()
        );
    }

    if (!file_in.is_open()) {
        throw std::runtime_error(
            "open_ifstream: failed to read: " + path.string()
        );
    }

    return file_in;

}

void cascade::helper::scan_csv_dim(
    fs::path path, int *m_rows_ptr, int *n_cols_ptr
) {

    std::ifstream file_in = open_ifstream(path, fs::path(".csv"));

    int m_rows = 0;
    int n_cols = 0;
    bool is_first_line = true;
    std::string temp_str;
    std::string line_string;
    std::stringstream line_stream;

    while (std::getline(file_in, line_string)) {

        ++m_rows;
        line_stream.clear();
        line_stream << line_string;

        if (is_first_line) {
        
            // Iterate over first row to count columns
            while (std::getline(line_stream, temp_str, ',')) { ++n_cols; }
            is_first_line = false;

        } else {

            // Ensure subsequent rows meet col count of first column
            int col_count = 0;
            while (std::getline(line_stream, temp_str, ',')) { ++col_count; }
            if (col_count != n_cols) {
                throw std::runtime_error(
                    "scan_csv_dim: error in: " + path.string() + "\n"
                    "row " + std::to_string(m_rows) +
                    " does not meet column size of " + std::to_string(n_cols)
                );
            }

        }
    
    }

    *m_rows_ptr = m_rows;
    *n_cols_ptr = n_cols;

}

bool cascade::helper::no_whitespace(std::string &str) {
    for (auto iter = str.cbegin(); iter != str.cend(); ++iter) {
        if (std::isspace(*iter)) { return false; }
    }
    return true;
};
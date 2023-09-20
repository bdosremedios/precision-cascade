#ifndef MATRIXREADER_H
#define MATRIXREADER_H

#include "Eigen/Dense"

#include "types/types.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using Eigen::Matrix;
using Eigen::Dynamic;

using std::string, std::to_string;
using std::ifstream, std::stringstream, std::getline;
using std::runtime_error;
using std::cout, std::endl;

template <template<typename> typename M, typename T>
class SpecializedReturner
{
    public: M<T> return_(M<T>);
};

template <typename T>
class SpecializedReturner<MatrixVector, T>
{
public:
    MatrixVector<T> return_(MatrixDense<T> mat) {
        if (mat.cols() == 1) {
            return mat;
        } else {
            throw runtime_error("Vector can only read one column csv.");
        }
    }
};

template <typename T>
class SpecializedReturner<MatrixDense, T>
{
    public: MatrixDense<T> return_(MatrixDense<T> mat) { return mat; }
};

template <typename T>
class SpecializedReturner<MatrixSparse, T>
{
    public: MatrixSparse<T> return_(MatrixDense<T> mat) { return mat.sparseView(); }
};

template <template<typename> typename M, typename T>
M<T> read_matrixCSV(string const &path)
{

    assert_valid_type_or_vec<M>();

    // Open given file
    ifstream file_in;
    file_in.open(path);

    // Check that file opening was successful
    if (file_in.is_open()) {

        MatrixDense<T> mat;
        int n_rows = 0;
        int n_cols = 0;
        
        string line_read_temp;
        stringstream line;
        bool is_first_line = true;
        int element_count = 1;

        while (getline(file_in, line_read_temp)) {

            // Read string into stringstream
            line.clear();
            line << line_read_temp;

            // Count the first line to ensure all subsequent rows match the element
            // count
            if (is_first_line) {
                
                ++n_rows;
                string temp_str;
                T temp_number;

                // Iterate over comma seperated line
                while (getline(line, temp_str, ',')) {
                    // Expand mat's column by one and set as new entry if entry is valid
                    try {
                        temp_number = static_cast<T>(stod(temp_str));
                    } catch (std::invalid_argument e) {
                        throw runtime_error(
                            "Error in: " + path + "\n" +
                            "Invalid argument in file, failed to convert to numeric"
                        );
                    }
                    mat.conservativeResize(n_rows, ++n_cols);
                    mat(0, n_cols-1) = temp_number;
                }
                is_first_line = false;

            // For subsequent rows expand by row and read in next line, throwing
            // errors if any row exceeds to does not meet the original element count
            } else {

                mat.conservativeResize(++n_rows, n_cols);
                string temp_str;
                T temp_number;
                int curr_col_count = 0;

                // Iterate over comma seperated line
                while (getline(line, temp_str, ',')) {
                    // Check that the next added element won't exceed the column count
                    if (curr_col_count < n_cols) {
                        // Expand mat's column by one and set as new entry if entry is valid
                        try {
                            temp_number = static_cast<T>(stod(temp_str));
                        } catch (std::invalid_argument e) {
                            throw runtime_error(
                                "Error in: " + path + "\n" +
                                "Invalid argument in file, failed to convert to numeric"
                            );
                        }
                        mat.coeffRef(n_rows-1, curr_col_count++) = temp_number;
                    } else {
                        throw runtime_error(
                            "Error in: " + path + "\n" +
                            "Row " + to_string(n_rows) + " exceeds column size of " + to_string(n_cols)
                        );
                    }
                }

                // Check that newly added row meets the column count
                if (curr_col_count < n_cols) {
                    throw runtime_error(
                        "Error in: " + path + "\n" +
                        "Row " + to_string(n_rows) + " does not meet column size of " + to_string(n_cols)
                    );
                }

            }
        }

        SpecializedReturner<M, T> sr;
        return sr.return_(mat);

    } else { throw runtime_error("Failed to read: " + path); }

} // end read_matrixCSV


#endif
// #include "Eigen/Dense"
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <iostream>
// #include "read_matrix/MatrixReader.h"

// using Eigen::Matrix, Eigen::Dynamic;
// using std::string, std::to_string;
// using std::ifstream, std::stringstream, std::getline;
// using std::runtime_error;
// using std::cout, std::endl;

// namespace mxread
// {

// Matrix<double, Dynamic, Dynamic> MatrixReader::read_file_d(string const & path) const {

//     // Open given file
//     ifstream file_in;
//     file_in.open(path);

//     // Check that file opening was successful
//     if (file_in.is_open()) {

//         // cout << "Reading Matrix: " + path << endl;
//         Matrix<double, Dynamic, Dynamic> M = Matrix<double, 0, 0>();
//         int n_rows = 0;
//         int n_cols = 0;
        
//         string line_read_temp;
//         stringstream line;
//         bool is_first_line = true;
//         int element_count = 1;

//         while (getline(file_in, line_read_temp)) {

//             // Read string into stringstream
//             line.clear();
//             line << line_read_temp;

//             // Count the first line to ensure all subsequent rows match the element
//             // count
//             if (is_first_line) {
                
//                 ++n_rows;
//                 string temp = "";
//                 double temp_d = 0;

//                 // Iterate over comma seperated line
//                 while (getline(line, temp, ',')) {

//                     // Expand M's column by one and set as new entry if entry is valid
//                     try {
//                         temp_d = stod(temp);
//                     } catch (std::invalid_argument e) {
//                         throw runtime_error(
//                             "Error in: " + path + "\n" + "Invalid argument in file, failed to convert to numeric"
//                         );
//                     }

//                     M.conservativeResize(n_rows, ++n_cols);
//                     M(0, n_cols-1) = temp_d;

//                 }

//                 is_first_line = false;

//             }
            
//             // For subsequent rows expand by row and read in next line, throwing
//             // errors if any row exceeds to does not meet the original element count
//             else {
                
//                 M.conservativeResize(++n_rows, n_cols);
//                 string temp = "";
//                 double temp_d = 0;
//                 int curr_col_count = 0;

//                 // Iterate over comma seperated line
//                 while (getline(line, temp, ',')) {

//                     // Check that the next added element won't exceed the column count
//                     if (curr_col_count < n_cols) {

//                         // Expand M's column by one and set as new entry if entry is valid
//                         try {
//                             temp_d = stod(temp);
//                         } catch (std::invalid_argument e) {
//                             throw runtime_error(
//                                 "Error in: " + path + "\n" + "Invalid argument in file, failed to convert to numeric"
//                             );
//                         }
//                         M(n_rows-1, curr_col_count++) = temp_d;

//                     } else {
//                         throw runtime_error(
//                             "Error in: " + path + "\n" +
//                             "Row " + to_string(n_rows) + " exceeds column size of " + to_string(n_cols)
//                             );
//                     }

//                 }

//                 // Check that newly added row meets the column count
//                 if (curr_col_count < n_cols) {
//                     throw runtime_error(
//                         "Error in: " + path + "\n" +
//                         "Row " + to_string(n_rows) + " does not meet column size of " + to_string(n_cols)
//                         );
//                 }

//             }

//         }

//         return M;

//     } else {

//         throw runtime_error("Failed to read: " + path);

//     }

// }

// }
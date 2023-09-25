#ifndef MATRIXWRITER_H
#define MATRIXWRITER_H

#include "Eigen/Dense"

#include <string>
#include <fstream>

using Eigen::MatrixXd;

using std::string, std::to_string;
using std::ofstream;
using std::runtime_error;

void write_matrixCSV(
    const MatrixXd &mat,
    const string save_path
) {

    ofstream file_out;
    file_out.open(save_path);

    if (file_out.is_open()) {

        for (int i=0; i<mat.rows(); ++i) {
            for (int j=0; j<mat.cols(); ++j) {
                file_out << mat(i, j);
            }
            file_out << "\n";
        }

        file_out.close();

    } else { throw runtime_error("Failed to open for write: " + save_path); }

} // end write_matrixCSV

#endif
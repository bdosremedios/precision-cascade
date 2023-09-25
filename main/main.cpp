#include "Eigen/Dense"
#include "Eigen/SparseCore"

#include "tools/MatrixWriter.h"

#include <filesystem>
#include <iostream>
#include <string>

int main() {

    string test_path = "/home/bdosre/dev/numerical_experimentation/test/test_dir_1/";

    std::filesystem::create_directories(test_path);

    MatrixXd test_mat = MatrixXd::Random(6, 5);
    write_matrixCSV(
        test_mat, test_path + "test.csv"
    );

    return 0;

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <string>

using read_matrix::read_matrix_csv;
using Eigen::MatrixXf;
using std::string;
using std::cout, std::endl;

class PGMRESTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 4*pow(2, -52); // Set as 4 times machines epsilon
        double convergence_tolerance = 1e-11;

};
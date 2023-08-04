#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include "test.h"

#include <string>
#include <iostream>

using Eigen::MatrixXd;

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;

class PGMRESTest: public TestBase;

TEST_F(PGMRESTest, TestLeftPreconditioning_3eigs) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv(solve_matrix_dir + "conv_diff_64.csv"));
    // GMRESSolveTestingMock<double>()

}
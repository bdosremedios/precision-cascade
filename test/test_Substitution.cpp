#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "tools/Substitution.h"

#include <string>
#include <iostream>

using read_matrix::read_matrix_csv;

using Eigen::Matrix;

using std::string;
using std::cout, std::endl;

class SubstitutionTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 4*pow(2, -52); // Set as 4 times machines epsilon

};

TEST_F(SubstitutionTest, TestBackwardSubstitution) {

    Matrix<double, Dynamic, Dynamic> U_tri = read_matrix_csv<double>(matrix_dir + "U_tri_90.csv");
    Matrix<double, Dynamic, 1> x_tri = read_matrix_csv<double>(matrix_dir + "x_tri_90.csv");
    Matrix<double, Dynamic, 1> b_tri = read_matrix_csv<double>(matrix_dir + "b_tri_90.csv");
    Matrix<double, Dynamic, 1> test_soln = back_substitution(U_tri, b_tri, 90);

    for (int i=0; i<90; ++i) {
        ASSERT_NEAR(test_soln[i], x_tri[i], double_tolerance);
    }
    ASSERT_NEAR((b_tri-U_tri*test_soln).norm(), 0, double_tolerance);

}
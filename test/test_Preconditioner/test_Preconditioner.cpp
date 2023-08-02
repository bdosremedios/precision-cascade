#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <string>
#include <iostream>
#include <cmath>

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;
using std::pow;

class PreconditionerTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 4*pow(2, -52); // Set as 4 times machines epsilon

};

TEST_F(PreconditionerTest, TestNoPreconditioner) {

    NoPreconditioner<double> no_precond;

    ASSERT_TRUE(no_precond.check_compatibility_left(1));
    ASSERT_TRUE(no_precond.check_compatibility_right(5));

    Matrix<double, Dynamic, 1> test_vec = Matrix<double, Dynamic, 1>::Random(64, 1);
    ASSERT_EQ(test_vec, no_precond.action_inv_M(test_vec));

}

TEST_F(PreconditionerTest, TestMatrixInverse) {

    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_inv_45.csv");
    Matrix<double, Dynamic, Dynamic> A_inv = read_matrix_csv<double>(matrix_dir + "Ainv_inv_45.csv");
    MatrixInverse<double> inv_precond(A_inv);

    // Check compatibility of with only 45
    ASSERT_TRUE(inv_precond.check_compatibility_left(45));
    ASSERT_TRUE(inv_precond.check_compatibility_right(45));
    ASSERT_FALSE(inv_precond.check_compatibility_left(6));
    ASSERT_FALSE(inv_precond.check_compatibility_right(6));
    ASSERT_FALSE(inv_precond.check_compatibility_left(100));
    ASSERT_FALSE(inv_precond.check_compatibility_right(100));

    Matrix<double, Dynamic, 1> orig_test_vec = Matrix<double, Dynamic, 1>::Random(45, 1);
    Matrix<double, Dynamic, 1> test_vec = A*orig_test_vec;

    test_vec = inv_precond.action_inv_M(test_vec);

    for (int i=0; i<45; ++i) {
        ASSERT_NEAR(orig_test_vec[i], test_vec[i], pow(10, -12));
    }

}
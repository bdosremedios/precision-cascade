#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <string>
#include <iostream>

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;

TEST(PreconditionerTest, TestNoPreconditioner) {

    NoPreconditioner<double> no_precond;

    ASSERT_TRUE(no_precond.check_compatibility_left(1));
    ASSERT_TRUE(no_precond.check_compatibility_left(5));

    Matrix<double, Dynamic, 1> test_vec;
    ASSERT_EQ(test_vec, no_precond.action_inv_M(test_vec));

}
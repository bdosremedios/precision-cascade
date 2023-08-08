#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <cmath>

using read_matrix::read_matrix_csv;

class ILUTest: public TestBase {};

TEST_F(ILUTest, TestStraightLU) {

    // Test that using a completely dense matrix one just gets a LU
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_7_dummy_backsub.csv"));
    ILU<double> ilu(A, u_dbl, u_dbl);
    Matrix<double, Dynamic, Dynamic> test = ilu.get_L()*ilu.get_U()-A;
    for (int i=0; i<7; ++i) {
        for (int j=0; j<7; ++j) {
            EXPECT_NEAR(test(i, j), 0, gamma(7, u_dbl));
        }
    }

}
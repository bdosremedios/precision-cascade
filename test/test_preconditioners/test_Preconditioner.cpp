#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class PreconditionerTest: public TestBase {};

TEST_F(PreconditionerTest, TestNoPreconditioner) {

    constexpr int n(64);
    NoPreconditioner<double> no_precond;

    ASSERT_TRUE(no_precond.check_compatibility_left(1));
    ASSERT_TRUE(no_precond.check_compatibility_right(5));

    Matrix<double, Dynamic, 1> test_vec(Matrix<double, Dynamic, 1>::Random(n, 1));
    ASSERT_EQ(test_vec, no_precond.action_inv_M(test_vec));

}

TEST_F(PreconditionerTest, TestMatrixInverse) {

    constexpr int n(45);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, Dynamic, Dynamic> A_inv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    MatrixInverse<double> inv_precond(A_inv);

    // Check compatibility of with only 45
    ASSERT_TRUE(inv_precond.check_compatibility_left(n));
    ASSERT_TRUE(inv_precond.check_compatibility_right(n));
    ASSERT_FALSE(inv_precond.check_compatibility_left(6));
    ASSERT_FALSE(inv_precond.check_compatibility_right(6));
    ASSERT_FALSE(inv_precond.check_compatibility_left(100));
    ASSERT_FALSE(inv_precond.check_compatibility_right(100));

    Matrix<double, Dynamic, 1> orig_test_vec(Matrix<double, Dynamic, 1>::Random(45, 1));
    Matrix<double, Dynamic, 1> test_vec = A*orig_test_vec;

    test_vec = inv_precond.action_inv_M(test_vec);

    for (int i=0; i<n; ++i) {
        ASSERT_NEAR(orig_test_vec[i], test_vec[i], dbl_error_acc);
    }

}
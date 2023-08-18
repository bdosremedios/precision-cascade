#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <cmath>
#include <float.h>

using read_matrix::read_matrix_csv;
using std::abs;

class ILUTest: public TestBase {};

int count_zeros(Matrix<double, Dynamic, Dynamic> A, double zero_tol) {

    int count = 0;
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs(A(i, j)) <= zero_tol) { ++count; }
        }
    }

    return count;

}

TEST_F(ILUTest, TestSquareCheck) {

    try {
        Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Ones(7, 5);
        ILU<double> ilu(A, u_dbl);
        FAIL();
    } catch (runtime_error e) {
        cout << e.what() << endl;
    }

}

TEST_F(ILUTest, TestCompatibilityCheck) {

    // Test that 7x7 matrix is only compatible with 7
    constexpr int n(7);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_7_dummy_backsub.csv"));
    ILU<double> ilu(A, u_dbl);
    EXPECT_TRUE(ilu.check_compatibility_left(n));
    EXPECT_TRUE(ilu.check_compatibility_right(n));
    EXPECT_FALSE(ilu.check_compatibility_left(n-4));
    EXPECT_FALSE(ilu.check_compatibility_right(n-3));
    EXPECT_FALSE(ilu.check_compatibility_left(n+3));
    EXPECT_FALSE(ilu.check_compatibility_right(n+2));

}

TEST_F(ILUTest, TestZeroDiagonalEntries) {

    constexpr int n(7);

    try {
        Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Identity(n, n);
        A(0, 0) = 0;
        ILU<double> ilu(A, u_dbl);
        FAIL();
    } catch (runtime_error e) {
        cout << e.what() << endl;
    }

    try {
        Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Identity(n, n);
        A(4, 4) = 0;
        ILU<double> ilu(A, u_dbl);
        FAIL();
    } catch (runtime_error e) {
        cout << e.what() << endl;
    }

}

TEST_F(ILUTest, TestCorrectDenseLU) {

    // Test that using a completely dense matrix one just gets a LU
    constexpr int n(8);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_A.csv"));
    ILU<double> ilu(A, u_dbl);
    Matrix<double, Dynamic, Dynamic> test = ilu.get_L()*ilu.get_U()-A;

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            ASSERT_NEAR(test(i, j), 0, dbl_error_acc);
        }
    }

    // Test correct L and U triangularity
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            ASSERT_NEAR(ilu.get_L()(i, j), 0, dbl_error_acc);
        }
    }
    for (int i=0; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(ilu.get_U()(i, j), 0, dbl_error_acc);
        }
    }


    // Test matching ILU to MATLAB for the dense matrix
    Matrix<double, n, n> L(read_matrix_csv<double>(solve_matrix_dir + "ilu_L.csv"));
    Matrix<double, n, n> U(read_matrix_csv<double>(solve_matrix_dir + "ilu_U.csv"));

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            ASSERT_NEAR(ilu.get_L()(i, j), L(i, j), dbl_error_acc);
            ASSERT_NEAR(ilu.get_U()(i, j), U(i, j), dbl_error_acc);
        }
    }

}

TEST_F(ILUTest, TestSparseILU0) {

    // Test using a sparse matrix one matches the sparsity pattern for zero-fill
    constexpr int n(8);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_A.csv"));
    ILU<double> ilu(A, u_dbl);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (A(i, j) == 0.) {
                ASSERT_EQ(ilu.get_L()(i, j), 0.);
                ASSERT_EQ(ilu.get_U()(i, j), 0.);
                ASSERT_NEAR(ilu.get_L()(i, j)*ilu.get_U()(i, j), 0, dbl_error_acc);
            }
        }
    }
    
    // Test correct L and U triangularity
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            ASSERT_NEAR(ilu.get_L()(i, j), 0, dbl_error_acc);
        }
    }
    for (int i=0; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(ilu.get_U()(i, j), 0, dbl_error_acc);
        }
    }

    // Test matching ILU to MATLAB for the sparse for zero-fill matrix
    Matrix<double, n, n> L(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_L.csv"));
    Matrix<double, n, n> U(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_U.csv"));

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            ASSERT_NEAR(ilu.get_L()(i, j), L(i, j), dbl_error_acc);
            ASSERT_NEAR(ilu.get_U()(i, j), U(i, j), dbl_error_acc);
        }
    }

}

TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0) {

    // Test ILU(0) and ILUT(0) [No Dropping] Give the same dense decomp
    constexpr int n(8);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_A.csv"));
    ILU<double> ilu0(A, u_dbl);
    ILU<double> ilut0(A, u_dbl, 0);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (A(i, j) == 0.) {
                ASSERT_EQ(ilu0.get_L()(i, j), ilut0.get_L()(i, j));
                ASSERT_EQ(ilu0.get_U()(i, j), ilut0.get_U()(i, j));
            }
        }
    }

}

TEST_F(ILUTest, TestILUTDropping) {

    constexpr int n(8);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_A.csv"));

    // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
    // all have correct form for L and U
    ILU<double> ilut0_01(A, u_dbl, 0.01);
    ILU<double> ilut0_1(A, u_dbl, 0.1);
    ILU<double> ilut0_2(A, u_dbl, 0.2);
    ILU<double> ilut0_5(A, u_dbl, 0.5);

    // Test correct L and U triangularity
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            ASSERT_NEAR(ilut0_01.get_L()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_1.get_L()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_2.get_L()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_5.get_L()(i, j), 0, dbl_error_acc);
        }
    }
    for (int i=0; i<n; ++i) {
        for (int j=0; j<i; ++j) {
            ASSERT_NEAR(ilut0_01.get_U()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_1.get_U()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_2.get_U()(i, j), 0, dbl_error_acc);
            ASSERT_NEAR(ilut0_5.get_U()(i, j), 0, dbl_error_acc);
        }
    }

    // Test that each lower threshold is better than the higher one
    EXPECT_LE((A-ilut0_01.get_L()*ilut0_01.get_U()).norm(),
              (A-ilut0_1.get_L()*ilut0_1.get_U()).norm());
    EXPECT_LE((A-ilut0_1.get_L()*ilut0_1.get_U()).norm(),
              (A-ilut0_2.get_L()*ilut0_2.get_U()).norm());
    EXPECT_LE((A-ilut0_2.get_L()*ilut0_2.get_U()).norm(),
              (A-ilut0_5.get_L()*ilut0_5.get_U()).norm());

    // Test that each higher threshold has more zeros
    EXPECT_LE(count_zeros(ilut0_01.get_L(), u_dbl),
              count_zeros(ilut0_1.get_L(), u_dbl));
    EXPECT_LE(count_zeros(ilut0_1.get_L(), u_dbl),
              count_zeros(ilut0_2.get_L(), u_dbl));
    EXPECT_LE(count_zeros(ilut0_2.get_L(), u_dbl),
              count_zeros(ilut0_5.get_L(), u_dbl));
    EXPECT_LE(count_zeros(ilut0_01.get_U(), u_dbl),
              count_zeros(ilut0_1.get_U(), u_dbl));
    EXPECT_LE(count_zeros(ilut0_1.get_U(), u_dbl),
              count_zeros(ilut0_2.get_U(), u_dbl));
    EXPECT_LE(count_zeros(ilut0_2.get_U(), u_dbl),
              count_zeros(ilut0_5.get_U(), u_dbl));

}

TEST_F(ILUTest, TestILUTDroppingLimits) {

    // Test that max dropping just gives the diagonal since everything else gets dropped
    constexpr int n(8);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_A.csv"));
    ILU<double> ilu_all_drop(A, u_dbl, DBL_MAX);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (i != j) {
                ASSERT_NEAR(ilu_all_drop.get_L()(i, j), 0., dbl_error_acc);
                ASSERT_NEAR(ilu_all_drop.get_U()(i, j), 0., dbl_error_acc);
            } else {
                ASSERT_NEAR(ilu_all_drop.get_L()(i, j), 1., dbl_error_acc);
                ASSERT_NEAR(ilu_all_drop.get_U()(i, j), A(i, j), dbl_error_acc);
            }
        }
    }

}
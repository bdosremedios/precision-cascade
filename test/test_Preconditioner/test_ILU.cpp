#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <cmath>
#include <float.h>

using read_matrix::read_matrix_csv;

class ILUTest: public TestBase {};

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
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_7_dummy_backsub.csv"));
    ILU<double> ilu(A, u_dbl);
    EXPECT_TRUE(ilu.check_compatibility_left(7));
    EXPECT_TRUE(ilu.check_compatibility_right(7));
    EXPECT_FALSE(ilu.check_compatibility_left(3));
    EXPECT_FALSE(ilu.check_compatibility_right(4));
    EXPECT_FALSE(ilu.check_compatibility_left(10));
    EXPECT_FALSE(ilu.check_compatibility_right(9));

}

TEST_F(ILUTest, TestZeroDiagonalEntries) {

    try {
        Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Identity(7, 7);
        A(0, 0) = 0;
        ILU<double> ilu(A, u_dbl);
        FAIL();
    } catch (runtime_error e) {
        cout << e.what() << endl;
    }

    try {
        Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Identity(7, 7);
        A(4, 4) = 0;
        ILU<double> ilu(A, u_dbl);
        FAIL();
    } catch (runtime_error e) {
        cout << e.what() << endl;
    }

}

TEST_F(ILUTest, TestCorrectDenseLU) {

    // Test that using a completely dense matrix one just gets a LU
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_A.csv"));
    ILU<double> ilu(A, u_dbl);
    Matrix<double, Dynamic, Dynamic> test = ilu.get_L()*ilu.get_U()-A;

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            EXPECT_NEAR(test(i, j), 0, dbl_error_acc);
        }
    }

    // Test matching ILU to MATLAB for the dense matrix
    Matrix<double, Dynamic, Dynamic> L(read_matrix_csv<double>(solve_matrix_dir + "ilu_L.csv"));
    Matrix<double, Dynamic, Dynamic> U(read_matrix_csv<double>(solve_matrix_dir + "ilu_U.csv"));

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            EXPECT_NEAR(ilu.get_L()(i, j), L(i, j), dbl_error_acc);
            EXPECT_NEAR(ilu.get_U()(i, j), U(i, j), dbl_error_acc);
        }
    }

}

TEST_F(ILUTest, TestSparseILU0) {

    // Test using a sparse matrix one matches the sparsity pattern for zero-fill
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_A.csv"));
    ILU<double> ilu(A, u_dbl);

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            if (A(i, j) == 0.) {
                EXPECT_EQ(ilu.get_L()(i, j), 0.);
                EXPECT_EQ(ilu.get_U()(i, j), 0.);
                EXPECT_NEAR(ilu.get_L()(i, j)*ilu.get_U()(i, j), 0, dbl_error_acc);
            }
        }
    }

    // Test matching ILU to MATLAB for the sparse for zero-fill matrix
    Matrix<double, Dynamic, Dynamic> L(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_L.csv"));
    Matrix<double, Dynamic, Dynamic> U(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_U.csv"));

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            EXPECT_NEAR(ilu.get_L()(i, j), L(i, j), dbl_error_acc);
            EXPECT_NEAR(ilu.get_U()(i, j), U(i, j), dbl_error_acc);
        }
    }

}

TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0) {

    // Test ILU(0) and ILUT(0) [No Dropping] Give the same dense decomp
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_A.csv"));
    ILU<double> ilu0(A, u_dbl);
    ILU<double> ilut0(A, u_dbl, 0);

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            if (A(i, j) == 0.) {
                EXPECT_EQ(ilu0.get_L()(i, j), ilut0.get_L()(i, j));
                EXPECT_EQ(ilu0.get_U()(i, j), ilut0.get_U()(i, j));
            }
        }
    }

}

TEST_F(ILUTest, TestILUTDropping) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "ilu_sparse_A.csv"));

    // Test MATLAB ilut with 0.01 drop tolerance matches to within double accumulation error
    ILU<double> ilut0_01(A, u_dbl, 0.01);
    Matrix<double, Dynamic, Dynamic> L_0_01(read_matrix_csv<double>(solve_matrix_dir + "ilut_0_01_sparse_L.csv"));
    Matrix<double, Dynamic, Dynamic> U_0_01(read_matrix_csv<double>(solve_matrix_dir + "ilut_0_01_sparse_U.csv"));

    Matrix<double, Dynamic, Dynamic> ilutAbar = ilut0_01.get_L()*ilut0_01.get_U();
    Matrix<double, Dynamic, Dynamic> matlabAbar = L_0_01*U_0_01;

    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
            if (A(i, j) == 0.) {
                EXPECT_NEAR(ilutAbar(i, j), matlabAbar(i, j), dbl_error_acc);
            }
        }
    }


}
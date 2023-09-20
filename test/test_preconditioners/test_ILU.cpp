#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILUTest: public TestBase
{
public:

    template<template <typename> typename M>
    void TestSquareCheck() {

        try {
            M<double> A = M<double>::Ones(7, 5);
            ILU<M, double> ilu(A, u_dbl);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

    template<template <typename> typename M>
    void TestCompatibilityCheck() {

        // Test that 7x7 matrix is only compatible with 7
        constexpr int n(7);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_7_dummy_backsub.csv");
        ILU<M, double> ilu(A, u_dbl);
        EXPECT_TRUE(ilu.check_compatibility_left(n));
        EXPECT_TRUE(ilu.check_compatibility_right(n));
        EXPECT_FALSE(ilu.check_compatibility_left(n-4));
        EXPECT_FALSE(ilu.check_compatibility_right(n-3));
        EXPECT_FALSE(ilu.check_compatibility_left(n+3));
        EXPECT_FALSE(ilu.check_compatibility_right(n+2));

    }

    template<template <typename> typename M>
    void TestZeroDiagonalEntries() {

        constexpr int n(7);

        try {
            M<double> A = M<double>::Identity(n, n);
            A.coeffRef(0, 0) = 0;
            ILU<M, double> ilu(A, u_dbl);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

        try {
            M<double> A = M<double>::Identity(n, n);
            A.coeffRef(4, 4) = 0;
            ILU<M, double> ilu(A, u_dbl);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

    template<template <typename> typename M>
    void TestCorrectDenseLU() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_A.csv");
        ILU<M, double> ilu(A, u_dbl);
        M<double> test = ilu.get_L()*ilu.get_U()-A;

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(test.coeff(i, j), 0, dbl_error_acc);
            }
        }

        // Test correct L and U triangularity
        for (int i=0; i<n; ++i) {
            for (int j=i+1; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), 0, dbl_error_acc);
            }
        }
        for (int i=0; i<n; ++i) {
            for (int j=0; j<i; ++j) {
                ASSERT_NEAR(ilu.get_U().coeff(i, j), 0, dbl_error_acc);
            }
        }

        // Test matching ILU to MATLAB for the dense matrix
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_L.csv");
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_U.csv");

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), L.coeff(i, j), dbl_error_acc);
                ASSERT_NEAR(ilu.get_U().coeff(i, j), U.coeff(i, j), dbl_error_acc);
            }
        }

    }
    
    template<template <typename> typename M>
    void TestSparseILU0() {

        // Test using a sparse matrix one matches the sparsity pattern for zero-fill
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_sparse_A.csv");
        ILU<M, double> ilu(A, u_dbl);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (A.coeff(i, j) == 0.) {
                    ASSERT_EQ(ilu.get_L().coeff(i, j), 0.);
                    ASSERT_EQ(ilu.get_U().coeff(i, j), 0.);
                    ASSERT_NEAR(ilu.get_L().coeff(i, j)*ilu.get_U().coeff(i, j), 0, dbl_error_acc);
                }
            }
        }
        
        // Test correct L and U triangularity
        for (int i=0; i<n; ++i) {
            for (int j=i+1; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), 0, dbl_error_acc);
            }
        }
        for (int i=0; i<n; ++i) {
            for (int j=0; j<i; ++j) {
                ASSERT_NEAR(ilu.get_U().coeff(i, j), 0, dbl_error_acc);
            }
        }

        // Test matching ILU to MATLAB for the sparse for zero-fill matrix
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_sparse_L.csv");
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_sparse_U.csv");

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), L.coeff(i, j), dbl_error_acc);
                ASSERT_NEAR(ilu.get_U().coeff(i, j), U.coeff(i, j), dbl_error_acc);
            }
        }

    }

    template<template <typename> typename M>
    void TestEquivalentILUTNoDropAndDenseILU0() {

        // Test ILU(0) and ILUT(0) [No Dropping] Give the same dense decomp
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_A.csv");
        ILU<M, double> ilu0(A, u_dbl);
        ILU<M, double> ilut0(A, u_dbl, 0);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (A.coeff(i, j) == 0.) {
                    ASSERT_EQ(ilu0.get_L().coeff(i, j), ilut0.get_L().coeff(i, j));
                    ASSERT_EQ(ilu0.get_U().coeff(i, j), ilut0.get_U().coeff(i, j));
                }
            }
        }

    }

    template<template <typename> typename M>
    void TestILUTDropping() {

        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_sparse_A.csv");

        // Check multiple rising thresholds ensuring that each ilu is closer to the matrix and that
        // all have correct form for L and U
        ILU<M, double> ilut0_01(A, u_dbl, 0.01);
        ILU<M, double> ilut0_1(A, u_dbl, 0.1);
        ILU<M, double> ilut0_2(A, u_dbl, 0.2);
        ILU<M, double> ilut0_5(A, u_dbl, 0.5);

        // Test correct L and U triangularity
        for (int i=0; i<n; ++i) {
            for (int j=i+1; j<n; ++j) {
                ASSERT_NEAR(ilut0_01.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_1.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_2.get_L().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_5.get_L().coeff(i, j), 0, dbl_error_acc);
            }
        }
        for (int i=0; i<n; ++i) {
            for (int j=0; j<i; ++j) {
                ASSERT_NEAR(ilut0_01.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_1.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_2.get_U().coeff(i, j), 0, dbl_error_acc);
                ASSERT_NEAR(ilut0_5.get_U().coeff(i, j), 0, dbl_error_acc);
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

    template<template <typename> typename M>
    void TestILUTDroppingLimits() {

        // Test that max dropping just gives the diagonal since everything else gets dropped
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "ilu_sparse_A.csv");
        ILU<M, double> ilu_all_drop(A, u_dbl, DBL_MAX);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (i != j) {
                    ASSERT_NEAR(ilu_all_drop.get_L().coeff(i, j), 0., dbl_error_acc);
                    ASSERT_NEAR(ilu_all_drop.get_U().coeff(i, j), 0., dbl_error_acc);
                } else {
                    ASSERT_NEAR(ilu_all_drop.get_L().coeff(i, j), 1., dbl_error_acc);
                    ASSERT_NEAR(ilu_all_drop.get_U().coeff(i, j), A.coeff(i, j), dbl_error_acc);
                }
            }
        }

    }

};

template<template <typename> typename M, typename T>
int count_zeros(M<T> A, double zero_tol) {

    int count = 0;
    for (int i=0; i<A.rows(); ++i) {
        for (int j=0; j<A.cols(); ++j) {
            if (abs(A.coeff(i, j)) <= zero_tol) { ++count; }
        }
    }
    return count;

}

TEST_F(ILUTest, TestSquareCheck_Both) {
    TestSquareCheck<MatrixDense>();
    TestSquareCheck<MatrixSparse>();
}

TEST_F(ILUTest, TestCompatibilityCheck_Both) {
    TestCompatibilityCheck<MatrixDense>();
    TestCompatibilityCheck<MatrixSparse>();
}

TEST_F(ILUTest, TestZeroDiagonalEntries_Both) {
    TestZeroDiagonalEntries<MatrixDense>();
    TestZeroDiagonalEntries<MatrixSparse>();
}

TEST_F(ILUTest, TestCorrectDenseLU_Dense) { TestCorrectDenseLU<MatrixDense>(); }
TEST_F(ILUTest, TestCorrectDenseLU_Sparse) { TestCorrectDenseLU<MatrixSparse>(); }

TEST_F(ILUTest, TestSparseILU0_Dense) { TestSparseILU0<MatrixDense>(); }
TEST_F(ILUTest, TestSparseILU0_Sparse) { TestSparseILU0<MatrixSparse>(); }

TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0_Dense) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
}
TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0_Sparse) {
    TestEquivalentILUTNoDropAndDenseILU0<MatrixSparse>();
}

TEST_F(ILUTest, TestILUTDropping_Dense) { TestILUTDropping<MatrixDense>(); }
TEST_F(ILUTest, TestILUTDropping_Sparse) { TestILUTDropping<MatrixSparse>(); }

TEST_F(ILUTest, TestILUTDroppingLimits_Dense) { TestILUTDroppingLimits<MatrixDense>(); }
TEST_F(ILUTest, TestILUTDroppingLimits_Sparse) { TestILUTDroppingLimits<MatrixSparse>(); }
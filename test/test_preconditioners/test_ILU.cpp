#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILUTest: public TestBase
{
public:

    template<template <typename> typename M>
    void TestSquareCheck() {

        try {
            M<double> A = M<double>::Ones(7, 5);
            ILU<M, double> ilu(A, u_dbl, false);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

    template<template <typename> typename M>
    void TestCompatibilityCheck() {

        // Test that 7x7 matrix is only compatible with 7
        constexpr int n(7);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("A_7_dummy_backsub.csv"));
        ILU<M, double> ilu(A, u_dbl, false);
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
            ILU<M, double> ilu(A, u_dbl, false);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

        try {
            M<double> A = M<double>::Identity(n, n);
            A.coeffRef(4, 4) = 0;
            ILU<M, double> ilu(A, u_dbl, false);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

    template<template <typename> typename M>
    void TestCorrectDenseLU() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, u_dbl, false);
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
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_L.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_U.csv"));

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), L.coeff(i, j), dbl_error_acc);
                ASSERT_NEAR(ilu.get_U().coeff(i, j), U.coeff(i, j), dbl_error_acc);
            }
        }

    }
    
    template<template <typename> typename M>
    void TestCorrectDenseLU_Pivoted() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, u_dbl, true);
        M<double> test = ilu.get_L()*ilu.get_U()-A*ilu.get_P();

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(test.coeff(i, j), 0, dbl_error_acc);
            }
        }

        // Test correct L and U triangularity and correct permutation matrix P
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
        M<double> P_squared = ilu.get_P()*(ilu.get_P().transpose());
        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == j) {
                    ASSERT_NEAR(P_squared.coeff(i, j), 1, dbl_error_acc);
                } else {
                    ASSERT_NEAR(P_squared.coeff(i, j), 0, dbl_error_acc);
                }
            }
        }

        // Test matching ILU to MATLAB for the dense matrix
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_L_pivot.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_U_pivot.csv"));
        M<double> P = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_P_pivot.csv"));

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_P().coeff(i, j), P.coeff(i, j), dbl_error_acc);
            }
        }

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), L.coeff(i, j), dbl_error_acc);
                ASSERT_NEAR(ilu.get_U().coeff(i, j), U.coeff(i, j), dbl_error_acc);
            }
        }

    }

    template<template <typename> typename M>
    void TestCorrectLUApplyInverseM() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, u_dbl, false);
        
        // Test matching ILU to MATLAB for the dense matrix
        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        for (int i=0; i<n; ++i) {
            ASSERT_NEAR(ilu.action_inv_M(A*test_vec)(i), test_vec(i), dbl_error_acc);
        }

    }

    template<template <typename> typename M>
    void TestCorrectLUApplyInverseM_Pivoted() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, u_dbl, true);
        
        // Test matching ILU to MATLAB for the dense matrix
        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        for (int i=0; i<n; ++i) {
            ASSERT_NEAR(ilu.action_inv_M(A*test_vec)(i), test_vec(i), dbl_error_acc);
        }

    }
    
    template<template <typename> typename M>
    void TestSparseILU0() {

        // Test using a sparse matrix one matches the sparsity pattern for zero-fill
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
        ILU<M, double> ilu(A, u_dbl, false);

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
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_L.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_U.csv"));

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
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu0(A, u_dbl, false);
        ILU<M, double> ilut0(A, u_dbl, 0.);

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
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));

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
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
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

    template<template <typename> typename M>
    void TestDoubleSingleHalfCast() {

        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu0_dbl(A, u_dbl, false);

        M<double> L_dbl = ilu0_dbl.get_L();
        M<double> U_dbl = ilu0_dbl.get_U();

        M<float> L_sgl = ilu0_dbl.template get_L_cast<float>();
        M<float> U_sgl = ilu0_dbl.template get_U_cast<float>();

        ILU<M, float> ilu0_sgl(L_sgl, U_sgl);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(static_cast<float>(L_dbl.coeff(i, j)), L_sgl.coeff(i, j));
                ASSERT_EQ(static_cast<float>(U_dbl.coeff(i, j)), U_sgl.coeff(i, j));
                ASSERT_EQ(ilu0_sgl.get_L().coeff(i, j), L_sgl.coeff(i, j));
                ASSERT_EQ(ilu0_sgl.get_U().coeff(i, j), U_sgl.coeff(i, j));
            }
        }

        M<half> L_hlf = ilu0_dbl.template get_L_cast<half>();
        M<half> U_hlf = ilu0_dbl.template get_U_cast<half>();

        ILU<M, half> ilu0_hlf(L_hlf, U_hlf);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(static_cast<half>(L_dbl.coeff(i, j)), L_hlf.coeff(i, j));
                ASSERT_EQ(static_cast<half>(U_dbl.coeff(i, j)), U_hlf.coeff(i, j));
                ASSERT_EQ(ilu0_hlf.get_L().coeff(i, j), L_hlf.coeff(i, j));
                ASSERT_EQ(ilu0_hlf.get_U().coeff(i, j), U_hlf.coeff(i, j));
            }
        }

    }

    
    template<template <typename> typename M>
    void TestILUPremadeLUErrorChecks() {

        try {
            M<double> L_not_sq = M<double>::Random(8, 7);
            M<double> U = M<double>::Random(8, 8);
            ILU<M, double> ilu(L_not_sq, U);
            FAIL();
        } catch (runtime_error e) { ; }

        try {
            M<double> L = M<double>::Random(8, 8);
            M<double> U_not_sq = M<double>::Random(6, 8);
            ILU<M, double> ilu(L, U_not_sq);
            FAIL();
        } catch (runtime_error e) { ; }

        try {
            M<double> L_not_match = M<double>::Random(8, 8);
            M<double> U_not_match = M<double>::Random(7, 7);
            ILU<M, double> ilu(L_not_match, U_not_match);
            FAIL();
        } catch (runtime_error e) { ; }

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

TEST_F(ILUTest, TestCorrectDenseLU_Pivoted_Dense) { TestCorrectDenseLU_Pivoted<MatrixDense>(); }
TEST_F(ILUTest, TestCorrectDenseLU_Pivoted_Sparse) { TestCorrectDenseLU_Pivoted<MatrixSparse>(); }

TEST_F(ILUTest, TestCorrectLUApplyInverseM_Dense) { TestCorrectLUApplyInverseM<MatrixDense>(); }
TEST_F(ILUTest, TestCorrectLUApplyInverseM_Sparse) { TestCorrectLUApplyInverseM<MatrixSparse>(); }

TEST_F(ILUTest, TestCorrectLUApplyInverseM_Pivoted_Dense) { TestCorrectLUApplyInverseM_Pivoted<MatrixDense>(); }
TEST_F(ILUTest, TestCorrectLUApplyInverseM_Pivoted_Sparse) { TestCorrectLUApplyInverseM_Pivoted<MatrixSparse>(); }

TEST_F(ILUTest, TestSparseILU0_Dense) { TestSparseILU0<MatrixDense>(); }
TEST_F(ILUTest, TestSparseILU0_Sparse) { TestSparseILU0<MatrixSparse>(); }

// TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0_Dense) {
//     TestEquivalentILUTNoDropAndDenseILU0<MatrixDense>();
// }
// TEST_F(ILUTest, TestEquivalentILUTNoDropAndDenseILU0_Sparse) {
//     TestEquivalentILUTNoDropAndDenseILU0<MatrixSparse>();
// }

// TEST_F(ILUTest, TestILUTDropping_Dense) { TestILUTDropping<MatrixDense>(); }
// TEST_F(ILUTest, TestILUTDropping_Sparse) { TestILUTDropping<MatrixSparse>(); }

// TEST_F(ILUTest, TestILUTDroppingLimits_Dense) { TestILUTDroppingLimits<MatrixDense>(); }
// TEST_F(ILUTest, TestILUTDroppingLimits_Sparse) { TestILUTDroppingLimits<MatrixSparse>(); }

TEST_F(ILUTest, TestDoubleSingleHalfCast_Dense) { TestDoubleSingleHalfCast<MatrixDense>(); }
TEST_F(ILUTest, TestDoubleSingleHalfCast_Sparse) { TestDoubleSingleHalfCast<MatrixSparse>(); }

TEST_F(ILUTest, TestILUPremadeLUErrorChecks_Dense) { TestILUPremadeLUErrorChecks<MatrixDense>(); }
TEST_F(ILUTest, TestILUPremadeLUErrorChecks_Sparse) { TestILUPremadeLUErrorChecks<MatrixSparse>(); }
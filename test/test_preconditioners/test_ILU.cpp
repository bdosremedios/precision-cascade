#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILU_Test: public TestBase
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
    void TestApplyInverseM() {

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
    void TestApplyInverseM_Pivoted() {

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
    void TestILUPremadeErrorChecks() {

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

};

TEST_F(ILU_Test, TestSquareCheck_Both) {
    TestSquareCheck<MatrixDense>();
    TestSquareCheck<MatrixSparse>();
}

TEST_F(ILU_Test, TestCompatibilityCheck_Both) {
    TestCompatibilityCheck<MatrixDense>();
    TestCompatibilityCheck<MatrixSparse>();
}

TEST_F(ILU_Test, TestZeroDiagonalEntries_Both) {
    TestZeroDiagonalEntries<MatrixDense>();
    TestZeroDiagonalEntries<MatrixSparse>();
}

TEST_F(ILU_Test, TestApplyInverseM_Dense) { TestApplyInverseM<MatrixDense>(); }
TEST_F(ILU_Test, TestApplyInverseM_Sparse) { TestApplyInverseM<MatrixSparse>(); }

TEST_F(ILU_Test, TestApplyInverseM_Pivoted_Dense) { TestApplyInverseM_Pivoted<MatrixDense>(); }
TEST_F(ILU_Test, TestApplyInverseM_Pivoted_Sparse) { TestApplyInverseM_Pivoted<MatrixSparse>(); }

TEST_F(ILU_Test, TestILUPremadeErrorChecks_Dense) { TestILUPremadeErrorChecks<MatrixDense>(); }
TEST_F(ILU_Test, TestILUPremadeErrorChecks_Sparse) { TestILUPremadeErrorChecks<MatrixSparse>(); }

TEST_F(ILU_Test, TestDoubleSingleHalfCast_Dense) { TestDoubleSingleHalfCast<MatrixDense>(); }
TEST_F(ILU_Test, TestDoubleSingleHalfCast_Sparse) { TestDoubleSingleHalfCast<MatrixSparse>(); }
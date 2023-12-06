#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILU_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestSquareCheck() {

        try {
            M<double> A = M<double>::Ones(7, 5);
            ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
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
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
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
            ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

        try {
            M<double> A = M<double>::Identity(n, n);
            A.coeffRef(4, 4) = 0;
            ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
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
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
        
        // Test matching ILU to MATLAB for the dense matrix
        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        ASSERT_VECTOR_NEAR(ilu.action_inv_M(A*test_vec), test_vec, dbl_error_acc);

    }

    template<template <typename> typename M>
    void TestApplyInverseM_Pivoted() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, Tol<double>::roundoff(), true);
        
        // Test matching ILU to MATLAB for the dense matrix
        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        ASSERT_VECTOR_NEAR(ilu.action_inv_M(A*test_vec), test_vec, dbl_error_acc);

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
        ILU<M, double> ilu0_dbl(A, Tol<double>::roundoff(), false);

        M<double> L_dbl = ilu0_dbl.get_L();
        M<double> U_dbl = ilu0_dbl.get_U();

        M<float> L_sgl = ilu0_dbl.template get_L_cast<float>();
        M<float> U_sgl = ilu0_dbl.template get_U_cast<float>();

        ILU<M, float> ilu0_sgl(L_sgl, U_sgl);

        ASSERT_MATRIX_EQ(L_dbl.template cast<float>(), L_sgl);
        ASSERT_MATRIX_EQ(U_dbl.template cast<float>(), U_sgl);
        ASSERT_MATRIX_EQ(ilu0_sgl.get_L(), L_sgl);
        ASSERT_MATRIX_EQ(ilu0_sgl.get_U(), U_sgl);

        M<half> L_hlf = ilu0_dbl.template get_L_cast<half>();
        M<half> U_hlf = ilu0_dbl.template get_U_cast<half>();

        ILU<M, half> ilu0_hlf(L_hlf, U_hlf);

        ASSERT_MATRIX_EQ(L_dbl.template cast<half>(), L_hlf);
        ASSERT_MATRIX_EQ(U_dbl.template cast<half>(), U_hlf);
        ASSERT_MATRIX_EQ(ilu0_hlf.get_L(), L_hlf);
        ASSERT_MATRIX_EQ(ilu0_hlf.get_U(), U_hlf);

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
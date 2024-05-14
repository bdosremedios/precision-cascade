#include "../test.h"

#include "preconditioners/implemented_preconditioners.h"

class ILUPreconditioner_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestSquareCheck() {

        auto try_non_square = []() { 
            M<double> A(M<double>::Ones(TestBase::bundle, 7, 5));
            ILUPreconditioner<M, double> ilu(A, false);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_non_square);

    }

    template <template <typename> typename M>
    void TestCompatibilityCheck() {

        // Test that 7x7 matrix is only compatible with 7
        constexpr int n(7);
        M<double> A(M<double>::Identity(TestBase::bundle, 7, 7));
        ILUPreconditioner<M, double> ilu(A, false);
        EXPECT_TRUE(ilu.check_compatibility_left(n));
        EXPECT_TRUE(ilu.check_compatibility_right(n));
        EXPECT_FALSE(ilu.check_compatibility_left(n-4));
        EXPECT_FALSE(ilu.check_compatibility_right(n-3));
        EXPECT_FALSE(ilu.check_compatibility_left(n+3));
        EXPECT_FALSE(ilu.check_compatibility_right(n+2));

    }

    template <template <typename> typename M>
    void TestZeroDiagonalEntries() {

        constexpr int n(7);

        auto try_ilu_zero_at_0_0 = [=]() {
            MatrixDense<double> A(MatrixDense<double>::Identity(TestBase::bundle, n, n));
            A.set_elem(0, 0, 0);
            ILUPreconditioner<M, double> ilu(M<double>(A), false);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_ilu_zero_at_0_0);

        auto try_ilu_zero_at_4_4 = [=]() {
            MatrixDense<double> A(MatrixDense<double>::Identity(TestBase::bundle, n, n));
            A.set_elem(4, 4, 0);
            ILUPreconditioner<M, double> ilu(M<double>(A), false);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_ilu_zero_at_4_4);

    }

    template<template <typename> typename M>
    void TestApplyInverseM() {

        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")));
        ILUPreconditioner<M, double> ilu(A, false); // Make dense LU

        // Test matching ILU to MATLAB for the dense matrix
        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_NEAR(ilu.action_inv_M(A*test_vec), test_vec, Tol<double>::dbl_ilu_elem_tol());

    }

    template <template <typename> typename M>
    void TestApplyInverseM_Pivoted() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")));
        ILUPreconditioner<M, double> ilu(A, true);
        
        // Test matching ILU to MATLAB for the dense matrix
        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_NEAR(ilu.action_inv_M(A*test_vec), test_vec, Tol<double>::dbl_ilu_elem_tol());

    }

    template <template <typename> typename M>
    void TestILUPremadeErrorChecks() {

        auto try_smaller_mismatch_col = []() {
            M<double> L_not_sq(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 8, 7));
            M<double> U(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 8, 8));
            ILUPreconditioner<M, double> ilu(L_not_sq, U);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_smaller_mismatch_col);

        auto try_smaller_mismatch_row = []() {
            M<double> L(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 8, 8));
            M<double> U_not_sq(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 6, 8));
            ILUPreconditioner<M, double> ilu(L, U_not_sq);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_smaller_mismatch_row);

        auto try_smaller_mismatch_both = []() {
            M<double> L_not_match(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 8, 8));
            M<double> U_not_match(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 7, 7));
            ILUPreconditioner<M, double> ilu(L_not_match, U_not_match);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_smaller_mismatch_both);

    }

    template <template <typename> typename M>
    void TestDoubleSingleHalfCast() {

        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")));
        ILUPreconditioner<M, double> ilu0_dbl(A, false);

        M<double> L_dbl(ilu0_dbl.get_L());
        M<double> U_dbl(ilu0_dbl.get_U());

        M<float> L_sgl(ilu0_dbl.template get_L_cast<float>());
        M<float> U_sgl(ilu0_dbl.template get_U_cast<float>());

        ILUPreconditioner<M, float> ilu0_sgl(L_sgl, U_sgl);

        ASSERT_MATRIX_EQ(L_dbl.template cast<float>(), L_sgl);
        ASSERT_MATRIX_EQ(U_dbl.template cast<float>(), U_sgl);
        ASSERT_MATRIX_EQ(ilu0_sgl.get_L(), L_sgl);
        ASSERT_MATRIX_EQ(ilu0_sgl.get_U(), U_sgl);

        M<half> L_hlf(ilu0_dbl.template get_L_cast<half>());
        M<half> U_hlf(ilu0_dbl.template get_U_cast<half>());

        ILUPreconditioner<M, half> ilu0_hlf(L_hlf, U_hlf);

        ASSERT_MATRIX_EQ(L_dbl.template cast<half>(), L_hlf);
        ASSERT_MATRIX_EQ(U_dbl.template cast<half>(), U_hlf);
        ASSERT_MATRIX_EQ(ilu0_hlf.get_L(), L_hlf);
        ASSERT_MATRIX_EQ(ilu0_hlf.get_U(), U_hlf);

    }

};

TEST_F(ILUPreconditioner_Test, TestSquareCheck_PRECONDITIONER) {
    TestSquareCheck<MatrixDense>();
    TestSquareCheck<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestCompatibilityCheck_PRECONDITIONER) {
    TestCompatibilityCheck<MatrixDense>();
    TestCompatibilityCheck<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestZeroDiagonalEntries_PRECONDITIONER) {
    TestZeroDiagonalEntries<MatrixDense>();
    TestZeroDiagonalEntries<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestApplyInverseM_PRECONDITIONER) {
    TestApplyInverseM<MatrixDense>();
    TestApplyInverseM<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestApplyInverseM_Pivoted_PRECONDITIONER) {
    TestApplyInverseM_Pivoted<MatrixDense>();
    TestApplyInverseM_Pivoted<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestILUPremadeErrorChecks_PRECONDITIONER) {
    TestILUPremadeErrorChecks<MatrixDense>();
    TestILUPremadeErrorChecks<NoFillMatrixSparse>();
}

TEST_F(ILUPreconditioner_Test, TestDoubleSingleHalfCast_PRECONDITIONER) {
    TestDoubleSingleHalfCast<MatrixDense>();
    TestDoubleSingleHalfCast<NoFillMatrixSparse>();
}
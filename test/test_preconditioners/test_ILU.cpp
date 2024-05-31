#include "../test.h"

#include "preconditioners/implemented_preconditioners.h"

class ILUPreconditioner_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestSquareCheck() {

        auto try_non_square = []() { 
            M<double> A(M<double>::Ones(TestBase::bundle, 7, 5));
            ILUPreconditioner<M, double> ilu(A);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_non_square);

    }

    template <template <typename> typename M>
    void TestCompatibilityCheck() {

        // Test that 7x7 matrix is only compatible with 7
        constexpr int n(7);
        M<double> A(M<double>::Identity(TestBase::bundle, 7, 7));
        ILUPreconditioner<M, double> ilu(A);
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
            M<double> A_cast(A);
            ILUPreconditioner<M, double> ilu(A_cast);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_ilu_zero_at_0_0);

        auto try_ilu_zero_at_4_4 = [=]() {
            MatrixDense<double> A(MatrixDense<double>::Identity(TestBase::bundle, n, n));
            A.set_elem(4, 4, 0);
            M<double> A_cast(A);
            ILUPreconditioner<M, double> ilu(A_cast);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_ilu_zero_at_4_4);

    }

    template<template <typename> typename M>
    void TestApplyInverseM() {

        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")));
        ILUPreconditioner<M, double> ilu(A); // Make dense LU

        // Test matching ILU to MATLAB for the dense matrix
        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_NEAR(ilu.action_inv_M(A*test_vec), test_vec, Tol<double>::dbl_ilu_elem_tol());

    }

    template <template <typename> typename M>
    void TestApplyInverseM_Pivoted() {

        // Test that using a completely dense matrix one just gets a LU
        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv")));
        ILUPreconditioner<M, double> ilu(A);
        
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

    template<template <typename> typename M>
    void TestILUPreconditionerCast() {

        constexpr int n(10);

        Vector<double> test_vec_dbl(Vector<double>::Random(TestBase::bundle, n));
        Vector<float> test_vec_sgl(Vector<float>::Random(TestBase::bundle, n));
        Vector<__half> test_vec_hlf(Vector<__half>::Random(TestBase::bundle, n));

        M<double> A_dbl(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        M<float> A_sgl(A_dbl.template cast<float>());
        M<__half> A_hlf(A_dbl.template cast<__half>());

        ILUPreconditioner<M, double> ilu0_dbl(A_dbl);
        ILUPreconditioner<M, float> ilu0_sgl(A_sgl);
        ILUPreconditioner<M, __half> ilu0_hlf(A_hlf);

        M<double> L_dbl_dbl = ilu0_dbl.get_L().template cast<double>();
        M<double> U_dbl_dbl = ilu0_dbl.get_U().template cast<double>();
        M<double> P_dbl_dbl = ilu0_dbl.get_P().template cast<double>();

        M<float> L_dbl_sgl = ilu0_dbl.get_L().template cast<float>();
        M<float> U_dbl_sgl = ilu0_dbl.get_U().template cast<float>();
        M<float> P_dbl_sgl = ilu0_dbl.get_P().template cast<float>();

        M<__half> L_dbl_hlf = ilu0_dbl.get_L().template cast<__half>();
        M<__half> U_dbl_hlf = ilu0_dbl.get_U().template cast<__half>();
        M<__half> P_dbl_hlf = ilu0_dbl.get_P().template cast<__half>();

        ILUPreconditioner<M, double> * ilu0_dbl_dbl_ptr = ilu0_dbl.cast_dbl_ptr();
        ILUPreconditioner<M, float> * ilu0_dbl_sgl_ptr = ilu0_dbl.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * ilu0_dbl_hlf_ptr = ilu0_dbl.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            ilu0_dbl_dbl_ptr->action_inv_M(test_vec_dbl),
            U_dbl_dbl.back_sub(L_dbl_dbl.frwd_sub(P_dbl_dbl*test_vec_dbl)),
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_dbl_sgl_ptr->action_inv_M(test_vec_sgl),
            U_dbl_sgl.back_sub(L_dbl_sgl.frwd_sub(P_dbl_sgl*test_vec_sgl)),
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_dbl_hlf_ptr->action_inv_M(test_vec_hlf),
            U_dbl_hlf.back_sub(L_dbl_hlf.frwd_sub(P_dbl_hlf*test_vec_hlf)),
            Tol<__half>::gamma_T(n)
        );

        delete ilu0_dbl_dbl_ptr;
        delete ilu0_dbl_sgl_ptr;
        delete ilu0_dbl_hlf_ptr;

        M<double> L_sgl_dbl = ilu0_sgl.get_L().template cast<double>();
        M<double> U_sgl_dbl = ilu0_sgl.get_U().template cast<double>();
        M<double> P_sgl_dbl = ilu0_sgl.get_P().template cast<double>();

        M<float> L_sgl_sgl = ilu0_sgl.get_L().template cast<float>();
        M<float> U_sgl_sgl = ilu0_sgl.get_U().template cast<float>();
        M<float> P_sgl_sgl = ilu0_sgl.get_P().template cast<float>();

        M<__half> L_sgl_hlf = ilu0_sgl.get_L().template cast<__half>();
        M<__half> U_sgl_hlf = ilu0_sgl.get_U().template cast<__half>();
        M<__half> P_sgl_hlf = ilu0_sgl.get_P().template cast<__half>();

        ILUPreconditioner<M, double> * ilu0_sgl_dbl_ptr = ilu0_sgl.cast_dbl_ptr();
        ILUPreconditioner<M, float> * ilu0_sgl_sgl_ptr = ilu0_sgl.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * ilu0_sgl_hlf_ptr = ilu0_sgl.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            ilu0_sgl_dbl_ptr->action_inv_M(test_vec_dbl),
            U_sgl_dbl.back_sub(L_sgl_dbl.frwd_sub(P_sgl_dbl*test_vec_dbl)),
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_sgl_sgl_ptr->action_inv_M(test_vec_sgl),
            U_sgl_sgl.back_sub(L_sgl_sgl.frwd_sub(P_sgl_sgl*test_vec_sgl)),
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_sgl_hlf_ptr->action_inv_M(test_vec_hlf),
            U_sgl_hlf.back_sub(L_sgl_hlf.frwd_sub(P_sgl_hlf*test_vec_hlf)),
            Tol<__half>::gamma_T(n)
        );

        delete ilu0_sgl_dbl_ptr;
        delete ilu0_sgl_sgl_ptr;
        delete ilu0_sgl_hlf_ptr;

        M<double> L_hlf_dbl = ilu0_hlf.get_L().template cast<double>();
        M<double> U_hlf_dbl = ilu0_hlf.get_U().template cast<double>();
        M<double> P_hlf_dbl = ilu0_hlf.get_P().template cast<double>();

        M<float> L_hlf_sgl = ilu0_hlf.get_L().template cast<float>();
        M<float> U_hlf_sgl = ilu0_hlf.get_U().template cast<float>();
        M<float> P_hlf_sgl = ilu0_hlf.get_P().template cast<float>();

        M<__half> L_hlf_hlf = ilu0_hlf.get_L().template cast<__half>();
        M<__half> U_hlf_hlf = ilu0_hlf.get_U().template cast<__half>();
        M<__half> P_hlf_hlf = ilu0_hlf.get_P().template cast<__half>();

        ILUPreconditioner<M, double> * ilu0_hlf_dbl_ptr = ilu0_hlf.cast_dbl_ptr();
        ILUPreconditioner<M, float> * ilu0_hlf_sgl_ptr = ilu0_hlf.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * ilu0_hlf_hlf_ptr = ilu0_hlf.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            ilu0_hlf_dbl_ptr->action_inv_M(test_vec_dbl),
            U_hlf_dbl.back_sub(L_hlf_dbl.frwd_sub(P_hlf_dbl*test_vec_dbl)),
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_hlf_sgl_ptr->action_inv_M(test_vec_sgl),
            U_hlf_sgl.back_sub(L_hlf_sgl.frwd_sub(P_hlf_sgl*test_vec_sgl)),
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            ilu0_hlf_hlf_ptr->action_inv_M(test_vec_hlf),
            U_hlf_hlf.back_sub(L_hlf_hlf.frwd_sub(P_hlf_hlf*test_vec_hlf)),
            Tol<__half>::gamma_T(n)
        );

        delete ilu0_hlf_dbl_ptr;
        delete ilu0_hlf_sgl_ptr;
        delete ilu0_hlf_hlf_ptr;

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

TEST_F(ILUPreconditioner_Test, TestILUPreconditionerCast_PRECONDITIONER) {
    TestILUPreconditionerCast<MatrixDense>();
    TestILUPreconditionerCast<NoFillMatrixSparse>();
}
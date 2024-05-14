#include "../test.h"

#include "preconditioners/implemented_preconditioners.h"

class ILU0_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestMatchesDenseLU() {

        // Test that using a completely dense matrix one just gets LU
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, false);
        M<double> test(
            MatrixDense<double>(ilu_precond.get_L())*MatrixDense<double>(ilu_precond.get_U()) -
            MatrixDense<double>(A)
        );

        ASSERT_MATRIX_ZERO(test, Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        M<double> L(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_L.csv"))
        );
        M<double> U(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_U.csv"))
        );

        ASSERT_MATRIX_NEAR(
            ilu_precond.get_L(),
            L,
            mat_max_mag(L)*Tol<double>::dbl_ilu_elem_tol()
        );
        ASSERT_MATRIX_NEAR(
            ilu_precond.get_U(),
            U,
            mat_max_mag(U)*Tol<double>::dbl_ilu_elem_tol()
        );

    }
    
    template<template <typename> typename M>
    void TestMatchesDenseLU_Pivoted() {

        // Test that using a completely dense matrix one just gets a pivoted LU
        constexpr int n(8);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, true);
        M<double> test(
            MatrixDense<double>(ilu_precond.get_L())*MatrixDense<double>(ilu_precond.get_U()) -
            MatrixDense<double>(ilu_precond.get_P())*MatrixDense<double>(A)
        );

        ASSERT_MATRIX_ZERO(test, Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        // Test validity of permutation matrix P
        M<double> P_squared(
            MatrixDense<double>(ilu_precond.get_P())*MatrixDense<double>(ilu_precond.get_P().transpose())
        );
        ASSERT_MATRIX_IDENTITY(P_squared, Tol<double>::dbl_ilu_elem_tol());

        M<double> L(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_L_pivot.csv"))
        );
        M<double> U(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_U_pivot.csv"))
        );
        M<double> P(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("ilu_P_pivot.csv"))
        );

        ASSERT_MATRIX_NEAR(ilu_precond.get_L(), L, Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_NEAR(ilu_precond.get_U(), U, Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_NEAR(ilu_precond.get_P(), P, Tol<double>::dbl_ilu_elem_tol());

    }
    
    template<template <typename> typename M>
    void TestMatchesSparseILU0() {

        // Test sparsity matches zero pattern for ILU0 on sparse A
        constexpr int n(8);
        M<double> A(read_matrixCSV<M, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, false);

        M<double> L(read_matrixCSV<M, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_L.csv"))
        );
        M<double> U(read_matrixCSV<M, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("ilu_sparse_U.csv"))
        );

        ASSERT_MATRIX_SAMESPARSITY(L, A, Tol<double>::roundoff());
        ASSERT_MATRIX_SAMESPARSITY(U, A, Tol<double>::roundoff());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        ASSERT_MATRIX_NEAR(ilu_precond.get_L(), L, Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_NEAR(ilu_precond.get_U(), U, Tol<double>::dbl_ilu_elem_tol());

    }

};

TEST_F(ILU0_Test, TestMatchesDenseLU_PRECONDITIONER) {
    TestMatchesDenseLU<MatrixDense>();
    TestMatchesDenseLU<NoFillMatrixSparse>();
}

TEST_F(ILU0_Test, TestMatchesDenseLU_Pivoted_PRECONDITIONER) {
    TestMatchesDenseLU_Pivoted<MatrixDense>();
    TestMatchesDenseLU_Pivoted<NoFillMatrixSparse>();
}

TEST_F(ILU0_Test, TestMatchesSparseILU0_PRECONDITIONER) {
    TestMatchesSparseILU0<MatrixDense>();
    TestMatchesSparseILU0<NoFillMatrixSparse>();
}
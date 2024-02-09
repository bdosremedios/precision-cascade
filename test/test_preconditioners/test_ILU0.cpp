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
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, Tol<double>::roundoff(), false);
        M<double> test(ilu_precond.get_L()*ilu_precond.get_U()-A);

        ASSERT_MATRIX_ZERO(test, Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        M<double> L(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_L.csv"))
        );
        M<double> U(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_U.csv"))
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
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, Tol<double>::roundoff(), true);
        M<double> test(ilu_precond.get_L()*ilu_precond.get_U()-ilu_precond.get_P()*A);

        ASSERT_MATRIX_ZERO(test, Tol<double>::dbl_ilu_elem_tol());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        // Test correct permutation matrix P
        M<double> P_squared(ilu_precond.get_P()*(ilu_precond.get_P().transpose()));
        ASSERT_MATRIX_IDENTITY(P_squared, Tol<double>::dbl_ilu_elem_tol());

        M<double> L(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_L_pivot.csv"))
        );
        M<double> U(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_U_pivot.csv"))
        );
        M<double> P(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("ilu_P_pivot.csv"))
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
            *handle_ptr, solve_matrix_dir / fs::path("ilu_sparse_A.csv"))
        );
        ILUPreconditioner<M, double> ilu_precond(A, Tol<double>::roundoff(), false);

        M<double> L(read_matrixCSV<M, double>(
            *handle_ptr, solve_matrix_dir / fs::path("ilu_sparse_L.csv"))
        );
        M<double> U(read_matrixCSV<M, double>(
            *handle_ptr, solve_matrix_dir / fs::path("ilu_sparse_U.csv"))
        );

        ASSERT_MATRIX_SAMESPARSITY(L, A, Tol<double>::roundoff());
        ASSERT_MATRIX_SAMESPARSITY(U, A, Tol<double>::roundoff());

        ASSERT_MATRIX_LOWTRI(ilu_precond.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu_precond.get_U(), Tol<double>::roundoff());

        ASSERT_MATRIX_NEAR(ilu_precond.get_L(), L, Tol<double>::dbl_ilu_elem_tol());
        ASSERT_MATRIX_NEAR(ilu_precond.get_U(), U, Tol<double>::dbl_ilu_elem_tol());

    }

};

TEST_F(ILU0_Test, TestMatchesDenseLU) {
    TestMatchesDenseLU<MatrixDense>();
    // TestMatchesDenseLU<MatrixSparse>();
}

TEST_F(ILU0_Test, TestMatchesDenseLU_Pivoted) {
    TestMatchesDenseLU_Pivoted<MatrixDense>();
    // TestMatchesDenseLU_Pivoted<MatrixSparse>();
}

TEST_F(ILU0_Test, TestMatchesSparseILU0) {
    TestMatchesSparseILU0<MatrixDense>();
    // TestMatchesSparseILU0<MatrixSparse>();
}
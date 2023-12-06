#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILU0_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestMatchesDenseLU() {

        // Test that using a completely dense matrix one just gets LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
        M<double> test = ilu.get_L()*ilu.get_U()-A;

        ASSERT_MATRIX_ZERO(test, dbl_error_acc);

        ASSERT_MATRIX_LOWTRI(ilu.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu.get_U(), Tol<double>::roundoff());

        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_L.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_U.csv"));

        ASSERT_MATRIX_NEAR(ilu.get_L(), L, dbl_error_acc);
        ASSERT_MATRIX_NEAR(ilu.get_U(), U, dbl_error_acc);

    }
    
    template<template <typename> typename M>
    void TestMatchesDenseLU_Pivoted() {

        // Test that using a completely dense matrix one just gets a pivoted LU
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_A.csv"));
        ILU<M, double> ilu(A, Tol<double>::roundoff(), true);
        M<double> test = ilu.get_L()*ilu.get_U()-ilu.get_P()*A;

        ASSERT_MATRIX_ZERO(test, dbl_error_acc);

        ASSERT_MATRIX_LOWTRI(ilu.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu.get_U(), Tol<double>::roundoff());

        // Test correct permutation matrix P
        M<double> P_squared = ilu.get_P()*(ilu.get_P().transpose());
        ASSERT_MATRIX_IDENTITY(P_squared, dbl_error_acc);

        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_L_pivot.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_U_pivot.csv"));
        M<double> P = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_P_pivot.csv"));

        ASSERT_MATRIX_NEAR(ilu.get_L(), L, dbl_error_acc);
        ASSERT_MATRIX_NEAR(ilu.get_U(), U, dbl_error_acc);
        ASSERT_MATRIX_NEAR(ilu.get_P(), P, dbl_error_acc);

    }
    
    template<template <typename> typename M>
    void TestMatchesSparseILU0() {

        // Test sparsity matches zero pattern for ILU0 on sparse A
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);

        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_L.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_U.csv"));

        ASSERT_MATRIX_SAMESPARSITY(L, A, Tol<double>::roundoff());
        ASSERT_MATRIX_SAMESPARSITY(U, A, Tol<double>::roundoff());

        ASSERT_MATRIX_LOWTRI(ilu.get_L(), Tol<double>::roundoff());
        ASSERT_MATRIX_UPPTRI(ilu.get_U(), Tol<double>::roundoff());

        ASSERT_MATRIX_NEAR(ilu.get_L(), L, dbl_error_acc);
        ASSERT_MATRIX_NEAR(ilu.get_U(), U, dbl_error_acc);

    }

};

TEST_F(ILU0_Test, TestMatchesDenseLU_Dense) { TestMatchesDenseLU<MatrixDense>(); }
TEST_F(ILU0_Test, TestMatchesDenseLU_Sparse) { TestMatchesDenseLU<MatrixSparse>(); }

TEST_F(ILU0_Test, TestMatchesDenseLU_Pivoted_Dense) { TestMatchesDenseLU_Pivoted<MatrixDense>(); }
TEST_F(ILU0_Test, TestMatchesDenseLU_Pivoted_Sparse) { TestMatchesDenseLU_Pivoted<MatrixSparse>(); }

TEST_F(ILU0_Test, TestMatchesSparseILU0_Dense) { TestMatchesSparseILU0<MatrixDense>(); }
TEST_F(ILU0_Test, TestMatchesSparseILU0_Sparse) { TestMatchesSparseILU0<MatrixSparse>(); }
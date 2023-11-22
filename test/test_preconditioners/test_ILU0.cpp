#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class ILU0_Test: public TestBase
{
public:

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
        M<double> test = ilu.get_L()*ilu.get_U()-ilu.get_P()*A;

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
    void TestSparseILU0() {

        // Test using a sparse matrix one matches the sparsity pattern for zero-fill
        constexpr int n(8);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_A.csv"));
        ILU<M, double> ilu(A, u_dbl, false);

        // Test matching ILU to MATLAB for the sparse for zero-fill matrix
        M<double> L = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_L.csv"));
        M<double> U = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("ilu_sparse_U.csv"));

        // A.print();
        // ilu.get_L().print();
        // ilu.get_U().print();

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                if (A.coeff(i, j) == 0.) {
                    ASSERT_EQ(ilu.get_L().coeff(i, j), 0.);
                    ASSERT_EQ(ilu.get_U().coeff(i, j), 0.);
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

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(ilu.get_L().coeff(i, j), L.coeff(i, j), dbl_error_acc);
                ASSERT_NEAR(ilu.get_U().coeff(i, j), U.coeff(i, j), dbl_error_acc);
            }
        }

    }

};

TEST_F(ILU0_Test, TestCorrectDenseLU_Dense) { TestCorrectDenseLU<MatrixDense>(); }
TEST_F(ILU0_Test, TestCorrectDenseLU_Sparse) { TestCorrectDenseLU<MatrixSparse>(); }

TEST_F(ILU0_Test, TestCorrectDenseLU_Pivoted_Dense) { TestCorrectDenseLU_Pivoted<MatrixDense>(); }
TEST_F(ILU0_Test, TestCorrectDenseLU_Pivoted_Sparse) { TestCorrectDenseLU_Pivoted<MatrixSparse>(); }

TEST_F(ILU0_Test, TestSparseILU0_Dense) { TestSparseILU0<MatrixDense>(); }
TEST_F(ILU0_Test, TestSparseILU0_Sparse) { TestSparseILU0<MatrixSparse>(); }
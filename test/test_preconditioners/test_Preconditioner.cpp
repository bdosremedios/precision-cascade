#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class Preconditioner_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestNoPreconditioner() {

        constexpr int n(64);
        NoPreconditioner<M, double> no_precond;

        ASSERT_TRUE(no_precond.check_compatibility_left(1));
        ASSERT_TRUE(no_precond.check_compatibility_right(5));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);
        ASSERT_VECTOR_EQ(no_precond.action_inv_M(test_vec), test_vec);
    
    }

    template<template <typename> typename M>
    void TestMatrixInverse() {
        
        constexpr int n(45);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
        M<double> A_inv = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
        MatrixInverse<M, double> inv_precond = A_inv;

        // Check compatibility with only 45
        ASSERT_TRUE(inv_precond.check_compatibility_left(n));
        ASSERT_TRUE(inv_precond.check_compatibility_right(n));
        ASSERT_FALSE(inv_precond.check_compatibility_left(6));
        ASSERT_FALSE(inv_precond.check_compatibility_right(6));
        ASSERT_FALSE(inv_precond.check_compatibility_left(100));
        ASSERT_FALSE(inv_precond.check_compatibility_right(100));

        MatrixVector<double> orig_test_vec = MatrixVector<double>::Random(n);
        MatrixVector<double> test_vec = A*orig_test_vec;

        test_vec = inv_precond.action_inv_M(test_vec);

        ASSERT_VECTOR_NEAR(orig_test_vec, test_vec, precond_error_tol);

    }

};

TEST_F(Preconditioner_Test, TestNoPreconditioner) {
    TestNoPreconditioner<MatrixDense>();
    TestNoPreconditioner<MatrixSparse>();
}

TEST_F(Preconditioner_Test, TestMatrixInverse) {
    TestMatrixInverse<MatrixDense>();
    TestMatrixInverse<MatrixSparse>();
}
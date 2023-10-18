#include "../test.h"

#include "preconditioners/ImplementedPreconditioners.h"

class PreconditionerTest: public TestBase
{
public:

    template<template <typename> typename M>
    void TestNoPreconditioner() {

        constexpr int n(64);
        NoPreconditioner<M, double> no_precond;

        ASSERT_TRUE(no_precond.check_compatibility_left(1));
        ASSERT_TRUE(no_precond.check_compatibility_right(5));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);
        ASSERT_EQ(test_vec, no_precond.action_inv_M(test_vec));
    
    }

    template<template <typename> typename M>
    void TestMatrixInverse() {
        
        constexpr int n(45);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
        M<double> A_inv = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
        MatrixInverse<M, double> inv_precond = A_inv;

        // Check compatibility of with only 45
        ASSERT_TRUE(inv_precond.check_compatibility_left(n));
        ASSERT_TRUE(inv_precond.check_compatibility_right(n));
        ASSERT_FALSE(inv_precond.check_compatibility_left(6));
        ASSERT_FALSE(inv_precond.check_compatibility_right(6));
        ASSERT_FALSE(inv_precond.check_compatibility_left(100));
        ASSERT_FALSE(inv_precond.check_compatibility_right(100));

        MatrixVector<double> orig_test_vec = MatrixVector<double>::Random(n);
        MatrixVector<double> test_vec = A*orig_test_vec;

        test_vec = inv_precond.action_inv_M(test_vec);

        for (int i=0; i<n; ++i) {
            ASSERT_NEAR(orig_test_vec[i], test_vec[i], dbl_error_acc);
        }

    }

};

TEST_F(PreconditionerTest, TestNoPreconditioner_Dense) { TestNoPreconditioner<MatrixDense>(); }
TEST_F(PreconditionerTest, TestNoPreconditioner_Sparse) { TestNoPreconditioner<MatrixSparse>(); }

TEST_F(PreconditionerTest, TestMatrixInverse_Dense) { TestMatrixInverse<MatrixDense>(); }
TEST_F(PreconditionerTest, TestMatrixInverse_Sparse) { TestMatrixInverse<MatrixSparse>(); }
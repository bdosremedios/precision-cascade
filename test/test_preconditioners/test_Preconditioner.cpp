#include "../test.h"

#include "preconditioners/implemented_preconditioners.h"

class Preconditioner_Test: public TestBase
{
public:

    template<template <typename> typename M>
    void TestNoPreconditioner() {

        constexpr int n(64);
        NoPreconditioner<M, double> no_precond;

        ASSERT_TRUE(no_precond.check_compatibility_left(1));
        ASSERT_TRUE(no_precond.check_compatibility_right(5));

        Vector<double> test_vec(Vector<double>::Random(*handle_ptr, n));
        ASSERT_VECTOR_EQ(no_precond.action_inv_M(test_vec), test_vec);
    
    }

    template<template <typename> typename M>
    void TestMatrixInversePreconditioner() {
        
        constexpr int n(45);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("A_inv_45.csv"))
        );
        MatrixInversePreconditioner<M, double> inv_precond(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("Ainv_inv_45.csv"))
        );

        // Check compatibility with only 45
        ASSERT_TRUE(inv_precond.check_compatibility_left(n));
        ASSERT_TRUE(inv_precond.check_compatibility_right(n));
        ASSERT_FALSE(inv_precond.check_compatibility_left(6));
        ASSERT_FALSE(inv_precond.check_compatibility_right(6));
        ASSERT_FALSE(inv_precond.check_compatibility_left(100));
        ASSERT_FALSE(inv_precond.check_compatibility_right(100));

        Vector<double> orig_test_vec(Vector<double>::Random(*handle_ptr, n));
        Vector<double> test_vec(A*orig_test_vec);
        test_vec = inv_precond.action_inv_M(test_vec);

        ASSERT_VECTOR_NEAR(orig_test_vec, test_vec, Tol<double>::inv_elem_tol());

    }

    template<template <typename> typename M>
    void TestCastInverseMAction() {
        
        constexpr int n(45);
        M<double> A(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("A_inv_45.csv"))
        );
        MatrixInversePreconditioner<M, double> inv_precond(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("Ainv_inv_45.csv"))
        );

        Vector<double> orig_test_vec(Vector<double>::Random(*handle_ptr, n));
        Vector<double> test_vec_dbl(inv_precond.action_inv_M(A*orig_test_vec));
        Vector<float> test_vec_sgl(inv_precond.template casted_action_inv_M<float>(A*orig_test_vec));

        ASSERT_VECTOR_NEAR(
            test_vec_sgl,
            test_vec_dbl.template cast<float>(),
            Tol<float>::inv_elem_tol()
        );

    }

};

TEST_F(Preconditioner_Test, TestNoPreconditioner) {
    TestNoPreconditioner<MatrixDense>();
    // TestNoPreconditioner<MatrixSparse>();
}

TEST_F(Preconditioner_Test, TestMatrixInversePreconditioner) {
    TestMatrixInversePreconditioner<MatrixDense>();
    // TestMatrixInverse<MatrixSparse>();
}

TEST_F(Preconditioner_Test, TestCastInverseMAction) {
    TestCastInverseMAction<MatrixDense>();
    // TestCastInverseMAction<MatrixSparse>();
}
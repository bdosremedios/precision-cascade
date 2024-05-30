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

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));
        ASSERT_VECTOR_EQ(no_precond.action_inv_M(test_vec), test_vec);
    
    }

    template<template <typename> typename M>
    void TestNoPreconditionerCast() {

        constexpr int n(64);

        Vector<double> test_vec_dbl(Vector<double>::Random(TestBase::bundle, n));
        Vector<float> test_vec_sgl(Vector<float>::Random(TestBase::bundle, n));
        Vector<__half> test_vec_hlf(Vector<__half>::Random(TestBase::bundle, n));

        NoPreconditioner<M, double> no_precond_dbl;
        NoPreconditioner<M, float> no_precond_sgl;
        NoPreconditioner<M, __half> no_precond_hlf;

        NoPreconditioner<M, double> * no_precond_dbl_dbl_ptr = no_precond_dbl.cast_dbl_ptr();
        NoPreconditioner<M, float> * no_precond_dbl_sgl_ptr = no_precond_dbl.cast_sgl_ptr();
        NoPreconditioner<M, __half> * no_precond_dbl_hlf_ptr = no_precond_dbl.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            no_precond_dbl_dbl_ptr->action_inv_M(test_vec_dbl),
            no_precond_dbl.action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_dbl_sgl_ptr->action_inv_M(test_vec_sgl),
            no_precond_sgl.action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_dbl_hlf_ptr->action_inv_M(test_vec_hlf),
            no_precond_hlf.action_inv_M(test_vec_hlf)
        );

        delete no_precond_dbl_dbl_ptr;
        delete no_precond_dbl_sgl_ptr;
        delete no_precond_dbl_hlf_ptr;

        NoPreconditioner<M, double> * no_precond_sgl_dbl_ptr = no_precond_sgl.cast_dbl_ptr();
        NoPreconditioner<M, float> * no_precond_sgl_sgl_ptr = no_precond_sgl.cast_sgl_ptr();
        NoPreconditioner<M, __half> * no_precond_sgl_hlf_ptr = no_precond_sgl.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            no_precond_sgl_dbl_ptr->action_inv_M(test_vec_dbl),
            no_precond_dbl.action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_sgl_sgl_ptr->action_inv_M(test_vec_sgl),
            no_precond_sgl.action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_sgl_hlf_ptr->action_inv_M(test_vec_hlf),
            no_precond_hlf.action_inv_M(test_vec_hlf)
        );

        delete no_precond_sgl_dbl_ptr;
        delete no_precond_sgl_sgl_ptr;
        delete no_precond_sgl_hlf_ptr;

        NoPreconditioner<M, double> * no_precond_hlf_dbl_ptr = no_precond_hlf.cast_dbl_ptr();
        NoPreconditioner<M, float> * no_precond_hlf_sgl_ptr = no_precond_hlf.cast_sgl_ptr();
        NoPreconditioner<M, __half> * no_precond_hlf_hlf_ptr = no_precond_hlf.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            no_precond_hlf_dbl_ptr->action_inv_M(test_vec_dbl),
            no_precond_dbl.action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_hlf_sgl_ptr->action_inv_M(test_vec_sgl),
            no_precond_sgl.action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            no_precond_hlf_hlf_ptr->action_inv_M(test_vec_hlf),
            no_precond_hlf.action_inv_M(test_vec_hlf)
        );

        delete no_precond_hlf_dbl_ptr;
        delete no_precond_hlf_sgl_ptr;
        delete no_precond_hlf_hlf_ptr;
    
    }

    template<template <typename> typename M>
    void TestMatrixInversePreconditioner() {
        
        constexpr int n(45);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_inv_45.csv"))
        );
        MatrixInversePreconditioner<M, double> inv_precond(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("Ainv_inv_45.csv"))
        );

        // Check compatibility with only 45
        ASSERT_TRUE(inv_precond.check_compatibility_left(n));
        ASSERT_TRUE(inv_precond.check_compatibility_right(n));
        ASSERT_FALSE(inv_precond.check_compatibility_left(6));
        ASSERT_FALSE(inv_precond.check_compatibility_right(6));
        ASSERT_FALSE(inv_precond.check_compatibility_left(100));
        ASSERT_FALSE(inv_precond.check_compatibility_right(100));

        Vector<double> orig_test_vec(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> test_vec(A*orig_test_vec);
        test_vec = inv_precond.action_inv_M(test_vec);

        ASSERT_VECTOR_NEAR(orig_test_vec, test_vec, Tol<double>::inv_elem_tol());

    }

    template<template <typename> typename M>
    void TestMatrixInverseCast() {

        constexpr int n(45);

        Vector<double> test_vec_dbl(Vector<double>::Random(TestBase::bundle, n));
        Vector<float> test_vec_sgl(Vector<float>::Random(TestBase::bundle, n));
        Vector<__half> test_vec_hlf(Vector<__half>::Random(TestBase::bundle, n));

        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_inv_45.csv"))
        );
        M<double> Ainv_dbl(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("Ainv_inv_45.csv"))
        );
        M<float> Ainv_sgl(Ainv_dbl.template cast<float>());
        M<__half> Ainv_hlf(Ainv_dbl.template cast<__half>());

        MatrixInversePreconditioner<M, double> inv_precond_dbl(Ainv_dbl);
        MatrixInversePreconditioner<M, float> inv_precond_sgl(Ainv_sgl);
        MatrixInversePreconditioner<M, __half> inv_precond_hlf(Ainv_hlf);

        MatrixInversePreconditioner<M, double> * inv_precond_dbl_dbl_ptr = inv_precond_dbl.cast_dbl_ptr();
        MatrixInversePreconditioner<M, float> * inv_precond_dbl_sgl_ptr = inv_precond_dbl.cast_sgl_ptr();
        MatrixInversePreconditioner<M, __half> * inv_precond_dbl_hlf_ptr = inv_precond_dbl.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            inv_precond_dbl_dbl_ptr->action_inv_M(test_vec_dbl),
            Ainv_dbl.template cast<double>()*test_vec_dbl,
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_dbl_sgl_ptr->action_inv_M(test_vec_sgl),
            Ainv_dbl.template cast<float>()*test_vec_sgl,
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_dbl_hlf_ptr->action_inv_M(test_vec_hlf),
            Ainv_dbl.template cast<__half>()*test_vec_hlf,
            Tol<__half>::gamma_T(n)
        );

        delete inv_precond_dbl_dbl_ptr;
        delete inv_precond_dbl_sgl_ptr;
        delete inv_precond_dbl_hlf_ptr;

        MatrixInversePreconditioner<M, double> * inv_precond_sgl_dbl_ptr = inv_precond_sgl.cast_dbl_ptr();
        MatrixInversePreconditioner<M, float> * inv_precond_sgl_sgl_ptr = inv_precond_sgl.cast_sgl_ptr();
        MatrixInversePreconditioner<M, __half> * inv_precond_sgl_hlf_ptr = inv_precond_sgl.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            inv_precond_sgl_dbl_ptr->action_inv_M(test_vec_dbl),
            Ainv_sgl.template cast<double>()*test_vec_dbl,
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_sgl_sgl_ptr->action_inv_M(test_vec_sgl),
            Ainv_sgl.template cast<float>()*test_vec_sgl,
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_sgl_hlf_ptr->action_inv_M(test_vec_hlf),
            Ainv_sgl.template cast<__half>()*test_vec_hlf,
            Tol<__half>::gamma_T(n)
        );

        delete inv_precond_sgl_dbl_ptr;
        delete inv_precond_sgl_sgl_ptr;
        delete inv_precond_sgl_hlf_ptr;

        MatrixInversePreconditioner<M, double> * inv_precond_hlf_dbl_ptr = inv_precond_hlf.cast_dbl_ptr();
        MatrixInversePreconditioner<M, float> * inv_precond_hlf_sgl_ptr = inv_precond_hlf.cast_sgl_ptr();
        MatrixInversePreconditioner<M, __half> * inv_precond_hlf_hlf_ptr = inv_precond_hlf.cast_hlf_ptr();

        ASSERT_VECTOR_NEAR(
            inv_precond_hlf_dbl_ptr->action_inv_M(test_vec_dbl),
            Ainv_hlf.template cast<double>()*test_vec_dbl,
            Tol<double>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_hlf_sgl_ptr->action_inv_M(test_vec_sgl),
            Ainv_hlf.template cast<float>()*test_vec_sgl,
            Tol<float>::gamma_T(n)
        );
        ASSERT_VECTOR_NEAR(
            inv_precond_hlf_hlf_ptr->action_inv_M(test_vec_hlf),
            Ainv_hlf.template cast<__half>()*test_vec_hlf,
            Tol<__half>::gamma_T(n)
        );

        delete inv_precond_hlf_dbl_ptr;
        delete inv_precond_hlf_sgl_ptr;
        delete inv_precond_hlf_hlf_ptr;
    
    }

};

TEST_F(Preconditioner_Test, TestNoPreconditioner_PRECONDITIONER) {
    TestNoPreconditioner<MatrixDense>();
    TestNoPreconditioner<NoFillMatrixSparse>();
}

TEST_F(Preconditioner_Test, TestNoPreconditionerCast_PRECONDITIONER) {
    TestNoPreconditionerCast<MatrixDense>();
    TestNoPreconditionerCast<NoFillMatrixSparse>();
}

TEST_F(Preconditioner_Test, TestMatrixInversePreconditioner_PRECONDITIONER) {
    TestMatrixInversePreconditioner<MatrixDense>();
    TestMatrixInversePreconditioner<NoFillMatrixSparse>();
}

TEST_F(Preconditioner_Test, TestMatrixInverseCast_PRECONDITIONER) {
    TestMatrixInverseCast<MatrixDense>();
    TestMatrixInverseCast<NoFillMatrixSparse>();
}
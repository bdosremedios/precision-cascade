#include "test.h"

#include "preconditioners/NoPreconditioner.h"

class NoPreconditioner_Test: public TestBase
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

};

TEST_F(NoPreconditioner_Test, TestNoPreconditioner_PRECONDITIONER) {
    TestNoPreconditioner<MatrixDense>();
    TestNoPreconditioner<NoFillMatrixSparse>();
}

TEST_F(NoPreconditioner_Test, TestNoPreconditionerCast_PRECONDITIONER) {
    TestNoPreconditionerCast<MatrixDense>();
    TestNoPreconditionerCast<NoFillMatrixSparse>();
}
#include "../../test.h"

#include "tools/arg_pkgs/PrecondArgPkg.h"

class PrecondArgPkg_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestDefaultConstruction() {

        constexpr int n(64);

        PrecondArgPkg<M, double> args;
        NoPreconditioner<M, double> no_precond;

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestLeftPreconditionerSet() {

        constexpr int n(14);

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        
        NoPreconditioner<M, double> no_precond;
        ILUPreconditioner<M, double> ilu_precond(A);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILUPreconditioner<M, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestRightPreconditionerSet() {

        constexpr int n(17);

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));

        NoPreconditioner<M, double> no_precond;
        ILUPreconditioner<M, double> ilu_precond(A);
        
        PrecondArgPkg<M, double> args(
            std::make_shared<NoPreconditioner<M, double>>(),
            std::make_shared<ILUPreconditioner<M, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));

        ILUPreconditioner<M, double> ilu_precond(A);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILUPreconditioner<M, double>>(A),
            std::make_shared<ILUPreconditioner<M, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestCastPtrs() {

        constexpr int n(12);

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));

        ILUPreconditioner<M, double> ilu_precond_dbl(A);
        ILUPreconditioner<M, float> ilu_precond_sgl(A.template cast<float>());
        ILUPreconditioner<M, __half> ilu_precond_hlf(A.template cast<__half>());

        PrecondArgPkg<M, double> args_dbl(
            std::make_shared<ILUPreconditioner<M, double>>(A),
            std::make_shared<ILUPreconditioner<M, double>>(A)
        );
        PrecondArgPkg<M, float> args_sgl(
            std::make_shared<ILUPreconditioner<M, float>>(A.template cast<float>()),
            std::make_shared<ILUPreconditioner<M, float>>(A.template cast<float>())
        );
        PrecondArgPkg<M, __half> args_hlf(
            std::make_shared<ILUPreconditioner<M, __half>>(A.template cast<__half>()),
            std::make_shared<ILUPreconditioner<M, __half>>(A.template cast<__half>())
        );

        Vector<double> test_vec_dbl(Vector<double>::Random(TestBase::bundle, n));
        Vector<float> test_vec_sgl(Vector<float>::Random(TestBase::bundle, n));
        Vector<__half> test_vec_hlf(Vector<__half>::Random(TestBase::bundle, n));

        PrecondArgPkg<M, double> * test_cast_dbl_dbl = args_dbl.cast_dbl_ptr();
        PrecondArgPkg<M, float> * test_cast_dbl_sgl = args_dbl.cast_sgl_ptr();
        PrecondArgPkg<M, __half> * test_cast_dbl_hlf = args_dbl.cast_hlf_ptr();

        ILUPreconditioner<M, double> * target_cast_dbl_dbl = ilu_precond_dbl.cast_dbl_ptr();
        ILUPreconditioner<M, float> * target_cast_dbl_sgl = ilu_precond_dbl.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * target_cast_dbl_hlf = ilu_precond_dbl.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            test_cast_dbl_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_dbl_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_dbl_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_dbl_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_dbl_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_dbl_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_dbl_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_dbl_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_dbl_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_dbl_hlf->action_inv_M(test_vec_hlf)
        );
        ASSERT_VECTOR_EQ(
            test_cast_dbl_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_dbl_hlf->action_inv_M(test_vec_hlf)
        );

        delete test_cast_dbl_dbl;
        delete test_cast_dbl_sgl;
        delete test_cast_dbl_hlf;

        delete target_cast_dbl_dbl;
        delete target_cast_dbl_sgl;
        delete target_cast_dbl_hlf;

        PrecondArgPkg<M, double> * test_cast_sgl_dbl = args_sgl.cast_dbl_ptr();
        PrecondArgPkg<M, float> * test_cast_sgl_sgl = args_sgl.cast_sgl_ptr();
        PrecondArgPkg<M, __half> * test_cast_sgl_hlf = args_sgl.cast_hlf_ptr();

        ILUPreconditioner<M, double> * target_cast_sgl_dbl = ilu_precond_sgl.cast_dbl_ptr();
        ILUPreconditioner<M, float> * target_cast_sgl_sgl = ilu_precond_sgl.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * target_cast_sgl_hlf = ilu_precond_sgl.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            test_cast_sgl_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_sgl_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_sgl_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_sgl_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_sgl_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_sgl_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_sgl_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_sgl_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_sgl_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_sgl_hlf->action_inv_M(test_vec_hlf)
        );
        ASSERT_VECTOR_EQ(
            test_cast_sgl_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_sgl_hlf->action_inv_M(test_vec_hlf)
        );

        delete test_cast_sgl_dbl;
        delete test_cast_sgl_sgl;
        delete test_cast_sgl_hlf;

        delete target_cast_sgl_dbl;
        delete target_cast_sgl_sgl;
        delete target_cast_sgl_hlf;

        PrecondArgPkg<M, double> * test_cast_hlf_dbl = args_hlf.cast_dbl_ptr();
        PrecondArgPkg<M, float> * test_cast_hlf_sgl = args_hlf.cast_sgl_ptr();
        PrecondArgPkg<M, __half> * test_cast_hlf_hlf = args_hlf.cast_hlf_ptr();

        ILUPreconditioner<M, double> * target_cast_hlf_dbl = ilu_precond_hlf.cast_dbl_ptr();
        ILUPreconditioner<M, float> * target_cast_hlf_sgl = ilu_precond_hlf.cast_sgl_ptr();
        ILUPreconditioner<M, __half> * target_cast_hlf_hlf = ilu_precond_hlf.cast_hlf_ptr();

        ASSERT_VECTOR_EQ(
            test_cast_hlf_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_hlf_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_hlf_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_hlf_dbl->action_inv_M(test_vec_dbl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_hlf_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_hlf_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_hlf_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_hlf_sgl->action_inv_M(test_vec_sgl)
        );
        ASSERT_VECTOR_EQ(
            test_cast_hlf_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_hlf_hlf->action_inv_M(test_vec_hlf)
        );
        ASSERT_VECTOR_EQ(
            test_cast_hlf_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_hlf_hlf->action_inv_M(test_vec_hlf)
        );

        delete test_cast_hlf_dbl;
        delete test_cast_hlf_sgl;
        delete test_cast_hlf_hlf;

        delete target_cast_hlf_dbl;
        delete target_cast_hlf_sgl;
        delete target_cast_hlf_hlf;

    }

};

TEST_F(PrecondArgPkg_Test, TestDefaultConstruction_PRECONDITIONER) {
    TestDefaultConstruction<MatrixDense>();
    TestDefaultConstruction<NoFillMatrixSparse>();
}

TEST_F(PrecondArgPkg_Test, TestLeftPreconditionerSet_PRECONDITIONER) {
    TestLeftPreconditionerSet<MatrixDense>();
    TestLeftPreconditionerSet<NoFillMatrixSparse>();
}

TEST_F(PrecondArgPkg_Test, TestRightPreconditionerSet_PRECONDITIONER) {
    TestRightPreconditionerSet<MatrixDense>();
    TestRightPreconditionerSet<NoFillMatrixSparse>();
}

TEST_F(PrecondArgPkg_Test, TestBothPreconditionerSet_PRECONDITIONER) {
    TestBothPreconditionerSet<MatrixDense>();
    TestBothPreconditionerSet<NoFillMatrixSparse>();
}

TEST_F(PrecondArgPkg_Test, TestCastPtrs_PRECONDITIONER) {
    TestCastPtrs<MatrixDense>();
    TestCastPtrs<NoFillMatrixSparse>();
}
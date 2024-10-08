#include "test.h"

#include "tools/arg_pkgs/PrecondArgPkg.h"

class PrecondArgPkg_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix>
    void TestDefaultConstruction() {

        constexpr int n(64);

        PrecondArgPkg<TMatrix, double> args;
        NoPreconditioner<TMatrix, double> no_precond;

        Vector<double> test_vec(
            Vector<double>::Random(TestBase::bundle, n)
        );

        ASSERT_VECTOR_EQ(
            args.left_precond->action_inv_M(test_vec),
            no_precond.action_inv_M(test_vec)
        );
        ASSERT_VECTOR_EQ(
            args.right_precond->action_inv_M(test_vec),
            no_precond.action_inv_M(test_vec)
        );

    }

    template <template <typename> typename TMatrix>
    void TestLeftPreconditionerSet() {

        constexpr int n(14);

        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, n, n
            )
        );
        
        NoPreconditioner<TMatrix, double> no_precond;
        ILUPreconditioner<TMatrix, double> ilu_precond(A);

        PrecondArgPkg<TMatrix, double> args(
            std::make_shared<ILUPreconditioner<TMatrix, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(
            args.left_precond->action_inv_M(test_vec),
            ilu_precond.action_inv_M(test_vec)
        );
        ASSERT_VECTOR_EQ(
            args.right_precond->action_inv_M(test_vec),
            no_precond.action_inv_M(test_vec)
        );

    }

    template <template <typename> typename TMatrix>
    void TestRightPreconditionerSet() {

        constexpr int n(17);

        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, n, n
            )
        );

        NoPreconditioner<TMatrix, double> no_precond;
        ILUPreconditioner<TMatrix, double> ilu_precond(A);
        
        PrecondArgPkg<TMatrix, double> args(
            std::make_shared<NoPreconditioner<TMatrix, double>>(),
            std::make_shared<ILUPreconditioner<TMatrix, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(
            args.left_precond->action_inv_M(test_vec),
            no_precond.action_inv_M(test_vec)
        );
        ASSERT_VECTOR_EQ(
            args.right_precond->action_inv_M(test_vec),
            ilu_precond.action_inv_M(test_vec)
        );

    }

    template <template <typename> typename TMatrix>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, n, n
            )
        );

        ILUPreconditioner<TMatrix, double> ilu_precond(A);

        PrecondArgPkg<TMatrix, double> args(
            std::make_shared<ILUPreconditioner<TMatrix, double>>(A),
            std::make_shared<ILUPreconditioner<TMatrix, double>>(A)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(
            args.left_precond->action_inv_M(test_vec),
            ilu_precond.action_inv_M(test_vec)
        );
        ASSERT_VECTOR_EQ(
            args.right_precond->action_inv_M(test_vec),
            ilu_precond.action_inv_M(test_vec)
        );

    }

    template <template <typename> typename TMatrix>
    void TestCastPtrs() {

        TMatrix<double> A_inv(read_matrixCSV<TMatrix, double>(
            TestBase::bundle,
            solve_matrix_dir / fs::path("Ainv_inv_45.csv")
        ));
        int n(A_inv.cols());

        MatrixInversePreconditioner<TMatrix, double> mat_inv_dbl(A_inv);
        MatrixInversePreconditioner<TMatrix, float> mat_inv_sgl(
            A_inv.template cast<float>()
        );
        MatrixInversePreconditioner<TMatrix, __half> mat_inv_hlf(
            A_inv.template cast<__half>()
        );

        PrecondArgPkg<TMatrix, double> args_dbl(
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                A_inv
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                A_inv
            )
        );
        PrecondArgPkg<TMatrix, float> args_sgl(
            std::make_shared<MatrixInversePreconditioner<TMatrix, float>>(
                A_inv.template cast<float>()
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, float>>(
                A_inv.template cast<float>()
            )
        );
        PrecondArgPkg<TMatrix, __half> args_hlf(
            std::make_shared<MatrixInversePreconditioner<TMatrix, __half>>(
                A_inv.template cast<__half>()
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, __half>>(
                A_inv.template cast<__half>()
            )
        );

        Vector<double> test_vec_dbl(
            Vector<double>::Random(TestBase::bundle, n)
        );
        Vector<float> test_vec_sgl(
            Vector<float>::Random(TestBase::bundle, n)
        );
        Vector<__half> test_vec_hlf(
            Vector<__half>::Random(TestBase::bundle, n)
        );

        PrecondArgPkg<TMatrix, double> * test_cast_dbl_dbl = (
            args_dbl.cast_dbl_ptr()
        );
        PrecondArgPkg<TMatrix, float> * test_cast_dbl_sgl = (
            args_dbl.cast_sgl_ptr()
        );
        PrecondArgPkg<TMatrix, __half> * test_cast_dbl_hlf = (
            args_dbl.cast_hlf_ptr()
        );

        MatrixInversePreconditioner<TMatrix, double> * target_cast_dbl_dbl = (
            mat_inv_dbl.cast_dbl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, float> * target_cast_dbl_sgl = (
            mat_inv_dbl.cast_sgl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, __half> * target_cast_dbl_hlf = (
            mat_inv_dbl.cast_hlf_ptr()
        );

        ASSERT_VECTOR_NEAR(
            test_cast_dbl_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_dbl_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_dbl_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_dbl_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_dbl_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_dbl_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_dbl_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_dbl_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_dbl_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_dbl_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_dbl_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_dbl_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
        );

        delete test_cast_dbl_dbl;
        delete test_cast_dbl_sgl;
        delete test_cast_dbl_hlf;

        delete target_cast_dbl_dbl;
        delete target_cast_dbl_sgl;
        delete target_cast_dbl_hlf;

        PrecondArgPkg<TMatrix, double> * test_cast_sgl_dbl = (
            args_sgl.cast_dbl_ptr()
        );
        PrecondArgPkg<TMatrix, float> * test_cast_sgl_sgl = (
            args_sgl.cast_sgl_ptr()
        );
        PrecondArgPkg<TMatrix, __half> * test_cast_sgl_hlf = (
            args_sgl.cast_hlf_ptr()
        );

        MatrixInversePreconditioner<TMatrix, double> * target_cast_sgl_dbl = (
            mat_inv_sgl.cast_dbl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, float> * target_cast_sgl_sgl = (
            mat_inv_sgl.cast_sgl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, __half> * target_cast_sgl_hlf = (
            mat_inv_sgl.cast_hlf_ptr()
        );

        ASSERT_VECTOR_NEAR(
            test_cast_sgl_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_sgl_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_sgl_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_sgl_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_sgl_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_sgl_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_sgl_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_sgl_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_sgl_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_sgl_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_sgl_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_sgl_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
        );

        delete test_cast_sgl_dbl;
        delete test_cast_sgl_sgl;
        delete test_cast_sgl_hlf;

        delete target_cast_sgl_dbl;
        delete target_cast_sgl_sgl;
        delete target_cast_sgl_hlf;

        PrecondArgPkg<TMatrix, double> * test_cast_hlf_dbl = (
            args_hlf.cast_dbl_ptr()
        );
        PrecondArgPkg<TMatrix, float> * test_cast_hlf_sgl = (
            args_hlf.cast_sgl_ptr()
        );
        PrecondArgPkg<TMatrix, __half> * test_cast_hlf_hlf = (
            args_hlf.cast_hlf_ptr()
        );

        MatrixInversePreconditioner<TMatrix, double> * target_cast_hlf_dbl = (
            mat_inv_hlf.cast_dbl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, float> * target_cast_hlf_sgl = (
            mat_inv_hlf.cast_sgl_ptr()
        );
        MatrixInversePreconditioner<TMatrix, __half> * target_cast_hlf_hlf = (
            mat_inv_hlf.cast_hlf_ptr()
        );

        ASSERT_VECTOR_NEAR(
            test_cast_hlf_dbl->left_precond->action_inv_M(test_vec_dbl),
            target_cast_hlf_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_hlf_dbl->right_precond->action_inv_M(test_vec_dbl),
            target_cast_hlf_dbl->action_inv_M(test_vec_dbl),
            Tol<double>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_hlf_sgl->left_precond->action_inv_M(test_vec_sgl),
            target_cast_hlf_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_hlf_sgl->right_precond->action_inv_M(test_vec_sgl),
            target_cast_hlf_sgl->action_inv_M(test_vec_sgl),
            Tol<float>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_hlf_hlf->left_precond->action_inv_M(test_vec_hlf),
            target_cast_hlf_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            test_cast_hlf_hlf->right_precond->action_inv_M(test_vec_hlf),
            target_cast_hlf_hlf->action_inv_M(test_vec_hlf),
            Tol<__half>::roundoff_T()
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
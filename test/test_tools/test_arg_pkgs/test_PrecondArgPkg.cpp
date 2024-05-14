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
        ILUPreconditioner<M, double> ilu_precond(A, false);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILUPreconditioner<M, double>>(A, false)
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
        ILUPreconditioner<M, double> ilu_precond(A, false);
        
        PrecondArgPkg<M, double> args(
            std::make_shared<NoPreconditioner<M, double>>(no_precond),
            std::make_shared<ILUPreconditioner<M, double>>(A, false)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));

        NoPreconditioner<M, double> no_precond;
        ILUPreconditioner<M, double> ilu_precond(A, false);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILUPreconditioner<M, double>>(A, false),
            std::make_shared<ILUPreconditioner<M, double>>(A, false)
        );

        Vector<double> test_vec(Vector<double>::Random(TestBase::bundle, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu_precond.action_inv_M(test_vec));

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
#include "../test.h"

#include "tools/PrecondArgPkg.h"

class PrecondArgPkg_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestDefaultConstruction() {

        constexpr int n(64);

        PrecondArgPkg<M, double> args;
        NoPreconditioner<M, double> no_precond;

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestLeftPreconditionerSet() {

        constexpr int n(14);

        M<double> A = M<double>::Random(n, n);
        
        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);

        PrecondArgPkg<M, double> args(make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestRightPreconditionerSet() {

        constexpr int n(17);

        M<double> A = M<double>::Random(n, n);

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
        
        PrecondArgPkg<M, double> args(make_shared<NoPreconditioner<M, double>>(no_precond),
                                      make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        M<double> A = M<double>::Random(n, n);

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);

        PrecondArgPkg<M, double> args(make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false),
                                      make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false));

        MatrixVector<double> test_vec(MatrixVector<double>::Random(n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

};

TEST_F(PrecondArgPkg_Test, TestDefaultConstruction_Both) {
    TestDefaultConstruction<MatrixDense>();
    TestDefaultConstruction<MatrixSparse>();
}

TEST_F(PrecondArgPkg_Test, TestLeftPreconditionerSet_Dense) { TestLeftPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkg_Test, TestLeftPreconditionerSet_Sparse) { TestLeftPreconditionerSet<MatrixSparse>(); }

TEST_F(PrecondArgPkg_Test, TestRightPreconditionerSet_Dense) { TestRightPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkg_Test, TestRightPreconditionerSet_Sparse) { TestRightPreconditionerSet<MatrixSparse>(); }

TEST_F(PrecondArgPkg_Test, TestBothPreconditionerSet_Dense) { TestBothPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkg_Test, TestBothPreconditionerSet_Sparse) { TestBothPreconditionerSet<MatrixSparse>(); }
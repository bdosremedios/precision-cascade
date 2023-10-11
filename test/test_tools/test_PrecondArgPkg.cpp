#include "../test.h"

#include "tools/PrecondArgPkg.h"

class PrecondArgPkgTest: public TestBase
{
public:

    template <template <typename> typename M>
    void TestDefaultConstruction() {

        constexpr int n(64);

        PrecondArgPkg<M, double> args;
        NoPreconditioner<M, double> no_precond;

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        EXPECT_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        EXPECT_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestLeftPreconditionerSet() {

        constexpr int n(14);

        M<double> A = M<double>::Random(n, n);
        
        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, u_dbl, false);

        PrecondArgPkg<M, double> args(make_shared<ILU<M, double>>(A, u_dbl, false));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        EXPECT_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        EXPECT_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestRightPreconditionerSet() {

        constexpr int n(17);

        M<double> A = M<double>::Random(n, n);

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, u_dbl, false);
        
        PrecondArgPkg<M, double> args(make_shared<NoPreconditioner<M, double>>(no_precond),
                                      make_shared<ILU<M, double>>(A, u_dbl, false));

        MatrixVector<double> test_vec = MatrixVector<double>::Random(n);

        EXPECT_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        EXPECT_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        M<double> A = M<double>::Random(n, n);

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, u_dbl, false);

        PrecondArgPkg<M, double> args(make_shared<ILU<M, double>>(A, u_dbl, false),
                                      make_shared<ILU<M, double>>(A, u_dbl, false));

        MatrixVector<double> test_vec(MatrixVector<double>::Random(n));

        EXPECT_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        EXPECT_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

};

TEST_F(PrecondArgPkgTest, TestDefaultConstruction_Both) {
    TestDefaultConstruction<MatrixDense>();
    TestDefaultConstruction<MatrixSparse>();
}

TEST_F(PrecondArgPkgTest, TestLeftPreconditionerSet_Dense) { TestLeftPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkgTest, TestLeftPreconditionerSet_Sparse) { TestLeftPreconditionerSet<MatrixSparse>(); }

TEST_F(PrecondArgPkgTest, TestRightPreconditionerSet_Dense) { TestRightPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkgTest, TestRightPreconditionerSet_Sparse) { TestRightPreconditionerSet<MatrixSparse>(); }

TEST_F(PrecondArgPkgTest, TestBothPreconditionerSet_Dense) { TestBothPreconditionerSet<MatrixDense>(); }
TEST_F(PrecondArgPkgTest, TestBothPreconditionerSet_Sparse) { TestBothPreconditionerSet<MatrixSparse>(); }
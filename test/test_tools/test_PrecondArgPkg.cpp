#include "../test.h"

#include "tools/ArgPkg.h"


class PrecondArgPkgTest: public TestBase {};

TEST_F(PrecondArgPkgTest, TestDefaultConstruction) {

    constexpr int n(64);

    PrecondArgPkg<double> args;
    NoPreconditioner<double> no_precond;

    Matrix<double, Dynamic, 1> test_vec(Matrix<double, Dynamic, 1>::Random(n, 1));

    EXPECT_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
    EXPECT_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

}

TEST_F(PrecondArgPkgTest, TestLeftPreconditionerSet) {

    constexpr int n(14);

    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(n, n);
    
    NoPreconditioner<double> no_precond;
    ILU<double> ilu(A, u_dbl);

    PrecondArgPkg<double> args(make_shared<ILU<double>>(A, u_dbl));

    Matrix<double, Dynamic, 1> test_vec(Matrix<double, Dynamic, 1>::Random(n, 1));

    EXPECT_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
    EXPECT_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

}

TEST_F(PrecondArgPkgTest, TestRightPreconditionerSet) {

    constexpr int n(17);

    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(n, n);

    NoPreconditioner<double> no_precond;
    ILU<double> ilu(A, u_dbl);
    
    PrecondArgPkg<double> args(make_shared<NoPreconditioner<double>>(no_precond),
                               make_shared<ILU<double>>(A, u_dbl));

    Matrix<double, Dynamic, 1> test_vec(Matrix<double, Dynamic, 1>::Random(n, 1));

    EXPECT_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
    EXPECT_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

}
TEST_F(PrecondArgPkgTest, TestBothPreconditionersSet) {

    constexpr int n(25);

    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(n, n);

    NoPreconditioner<double> no_precond;
    ILU<double> ilu(A, u_dbl);
    
    PrecondArgPkg<double> args(make_shared<ILU<double>>(A, u_dbl),
                               make_shared<ILU<double>>(A, u_dbl));

    Matrix<double, Dynamic, 1> test_vec(Matrix<double, Dynamic, 1>::Random(n, 1));

    EXPECT_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
    EXPECT_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

}
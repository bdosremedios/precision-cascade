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

        MatrixVector<double> test_vec(MatrixVector<double>::Random(*handle_ptr, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestLeftPreconditionerSet() {

        constexpr int n(14);

        M<double> A(M<double>::Random(*handle_ptr, n, n));
        
        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false)
        );

        MatrixVector<double> test_vec(MatrixVector<double>::Random(*handle_ptr, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestRightPreconditionerSet() {

        constexpr int n(17);

        M<double> A(M<double>::Random(*handle_ptr, n, n));

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);
        
        PrecondArgPkg<M, double> args(
            std::make_shared<NoPreconditioner<M, double>>(no_precond),
            std::make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false)
        );

        MatrixVector<double> test_vec(MatrixVector<double>::Random(*handle_ptr, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), no_precond.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

    template <template <typename> typename M>
    void TestBothPreconditionerSet() {

        constexpr int n(25);

        M<double> A(M<double>::Random(*handle_ptr, n, n));

        NoPreconditioner<M, double> no_precond;
        ILU<M, double> ilu(A, Tol<double>::roundoff(), false);

        PrecondArgPkg<M, double> args(
            std::make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false),
            std::make_shared<ILU<M, double>>(A, Tol<double>::roundoff(), false)
        );

        MatrixVector<double> test_vec(MatrixVector<double>::Random(*handle_ptr, n));

        ASSERT_VECTOR_EQ(args.left_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));
        ASSERT_VECTOR_EQ(args.right_precond->action_inv_M(test_vec), ilu.action_inv_M(test_vec));

    }

};

TEST_F(PrecondArgPkg_Test, TestDefaultConstruction) {
    TestDefaultConstruction<MatrixDense>();
    // TestDefaultConstruction<MatrixSparse>();
}

// TEST_F(PrecondArgPkg_Test, TestLeftPreconditionerSet) {
//     TestLeftPreconditionerSet<MatrixDense>();
//     // TestLeftPreconditionerSet<MatrixSparse>();
// }

// TEST_F(PrecondArgPkg_Test, TestRightPreconditionerSet) {
//     TestRightPreconditionerSet<MatrixDense>();
//     // TestRightPreconditionerSet<MatrixSparse>();
// }

// TEST_F(PrecondArgPkg_Test, TestBothPreconditionerSet) {
//     TestBothPreconditionerSet<MatrixDense>();
//     // TestBothPreconditionerSet<MatrixSparse>();
// }
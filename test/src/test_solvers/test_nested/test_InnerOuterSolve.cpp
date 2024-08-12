#include "test_InnerOuterSolve.h"

class Test_InnerOuterSolve: public TestBase
{
public:

    const int n = 16;

    template <template <typename> typename TMatrix>
    void Test_Initialization() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(1, 1, Tol<double>::roundoff()*1e2, init_guess);

        InnerOuterSolve_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        ASSERT_TRUE(solver.init_inner_outer_hit);
        ASSERT_EQ(solver.get_inner_res_norm_hist().size(), 0);

    }

    template <template <typename> typename TMatrix>
    void Test_Iterate() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(5, 3, Tol<double>::roundoff()*1e2, init_guess);

        InnerOuterSolve_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 0);
        ASSERT_EQ(solver.outer_iterate_complete_hit_count, 0);

        solver.iterate();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 1);
        ASSERT_EQ(solver.outer_iterate_complete_hit_count, 1);
        ASSERT_EQ(solver.get_inner_res_norm_hist().size(), 1);
        ASSERT_EQ(solver.get_inner_res_norm_hist()[0].size(), 4);
        ASSERT_EQ(solver.get_inner_iterations().size(), 1);
        ASSERT_EQ(solver.get_inner_iterations()[0], 3);

    }

    template <template <typename> typename TMatrix>
    void Test_Solve() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(5, 3, Tol<double>::roundoff()*1e2, init_guess);

        InnerOuterSolve_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        solver.solve();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 5);
        ASSERT_EQ(solver.outer_iterate_complete_hit_count, 5);
        ASSERT_EQ(solver.get_inner_res_norm_hist().size(), 5);
        for (int i=0; i<5; ++i) {
            ASSERT_EQ(solver.get_inner_res_norm_hist()[i].size(), 4);
        }
        ASSERT_EQ(solver.get_inner_iterations().size(), 5);
        for (int i=0; i<5; ++i) {
            ASSERT_EQ(solver.get_inner_iterations()[i], 3);
        }

        ASSERT_TRUE(solver.check_converged());

    }

    template <template <typename> typename TMatrix>
    void Test_SolveEarlyEnd() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(
            A * (
                init_guess +
                soln_add * Scalar<double>(
                    static_cast<double>(14.) / static_cast<double>(15.)
                )
            )
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(5, 3, Tol<double>::roundoff()*1e2, init_guess);

        InnerOuterSolve_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        solver.solve();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 5);
        ASSERT_EQ(solver.outer_iterate_complete_hit_count, 5);

        ASSERT_EQ(solver.get_inner_res_norm_hist().size(), 5);
        for (int i=0; i<4; ++i) {
            ASSERT_EQ(solver.get_inner_res_norm_hist()[i].size(), 4);
        }
        ASSERT_EQ(solver.get_inner_res_norm_hist()[4].size(), 3);

        ASSERT_EQ(solver.get_inner_iterations().size(), 5);
        for (int i=0; i<4; ++i) {
            ASSERT_EQ(solver.get_inner_iterations()[i], 3);
        }
        ASSERT_EQ(solver.get_inner_iterations()[4], 2);

    }

    template <template <typename> typename TMatrix>
    void Test_Reset() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(5, 3, Tol<double>::roundoff()*1e2, init_guess);

        InnerOuterSolve_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        solver.solve();
        solver.reset();

        ASSERT_EQ(solver.get_inner_res_norm_hist().size(), 0);
        ASSERT_EQ(solver.get_inner_iterations().size(), 0);

    }

};

TEST_F(Test_InnerOuterSolve, Test_Initialization) {
    Test_Initialization<MatrixDense>();
    Test_Initialization<NoFillMatrixSparse>();
}

TEST_F(Test_InnerOuterSolve, Test_Iterate) {
    Test_Iterate<MatrixDense>();
    Test_Iterate<NoFillMatrixSparse>();
}

TEST_F(Test_InnerOuterSolve, Test_Solve) {
    Test_Solve<MatrixDense>();
    Test_Solve<NoFillMatrixSparse>();
}

TEST_F(Test_InnerOuterSolve, Test_SolveEarlyEnd) {
    Test_SolveEarlyEnd<MatrixDense>();
    Test_SolveEarlyEnd<NoFillMatrixSparse>();
}

TEST_F(Test_InnerOuterSolve, Test_Reset) {
    Test_Reset<MatrixDense>();
    Test_Reset<NoFillMatrixSparse>();
}
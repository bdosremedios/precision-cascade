#include "test_IterativeRefinementBase.h"

class Test_IterativeRefinementBase: public TestBase
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

        IterativeRefinementBase_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        ASSERT_TRUE(solver.init_inner_outer_hit);
        ASSERT_EQ(solver.get_inner_res_norm_history().size(), 0);

    }

    template <template <typename> typename TMatrix>
    void Test_Iterate() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(2, 2, Tol<double>::roundoff()*1e2, init_guess);

        IterativeRefinementBase_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 0);

        solver.iterate();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 1);
        ASSERT_EQ(solver.get_inner_res_norm_history().size(), 1);
        ASSERT_EQ(solver.get_inner_res_norm_history()[0].size(), 3);
        ASSERT_EQ(solver.get_inner_iterations().size(), 1);
        ASSERT_EQ(solver.get_inner_iterations()[0], 2);

        Vector<double> target_soln(init_guess + (soln_add * Scalar<double>(0.5)));
        ASSERT_TRUE(solver.get_generic_soln() == target_soln);


    }

    template <template <typename> typename TMatrix>
    void Test_Solve() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(2, 2, Tol<double>::roundoff()*1e2, init_guess);

        IterativeRefinementBase_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        solver.solve();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 2);
        ASSERT_EQ(solver.get_inner_res_norm_history().size(), 2);
        for (int i=0; i<2; ++i) {
            ASSERT_EQ(solver.get_inner_res_norm_history()[i].size(), 3);
        }
        ASSERT_EQ(solver.get_inner_iterations().size(), 2);
        for (int i=0; i<2; ++i) {
            ASSERT_EQ(solver.get_inner_iterations()[i], 2);
        }

        ASSERT_TRUE(solver.check_converged());

    }

    template <template <typename> typename TMatrix>
    void Test_NaN() {

        TMatrix<double> A(MatrixDense<double>::Random(TestBase::bundle, n, n));
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> soln_add(Vector<double>::Random(TestBase::bundle, n));
        Vector<double> b(A*(init_guess + soln_add));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        SolveArgPkg args(2, 2, Tol<double>::roundoff()*1e2, init_guess);

        IterativeRefinementBase_Mock<TMatrix> solver(soln_add, &gen_lin_sys, args);

        solver.iterate();

        ASSERT_EQ(solver.outer_iterate_setup_hit_count, 1);
        ASSERT_EQ(solver.get_inner_res_norm_history().size(), 1);
        ASSERT_EQ(solver.get_inner_res_norm_history()[0].size(), 3);
        ASSERT_EQ(solver.get_inner_iterations().size(), 1);
        ASSERT_EQ(solver.get_inner_iterations()[0], 2);

        Vector<double> saved_soln = solver.get_generic_soln();

        solver.make_next_iterate_nan();
        solver.iterate();

        ASSERT_EQ(solver.get_inner_res_norm_history().size(), 2);

        std::vector<double> spun_vec = solver.get_inner_res_norm_history()[1];
        ASSERT_EQ(spun_vec.size(), 2);
        for (int i=0; i<spun_vec.size(); i++) {
            ASSERT_EQ(spun_vec[i], spun_vec[0]);
        }

        ASSERT_EQ(solver.get_inner_iterations().size(), 2);
        ASSERT_EQ(solver.get_inner_iterations()[1], 1);

        ASSERT_TRUE(saved_soln == solver.get_generic_soln());

    }

};

TEST_F(Test_IterativeRefinementBase, Test_Initialization) {
    Test_Initialization<MatrixDense>();
    Test_Initialization<NoFillMatrixSparse>();
}

TEST_F(Test_IterativeRefinementBase, Test_Iterate) {
    Test_Iterate<MatrixDense>();
    Test_Iterate<NoFillMatrixSparse>();
}

TEST_F(Test_IterativeRefinementBase, Test_Solve) {
    Test_Solve<MatrixDense>();
    Test_Solve<NoFillMatrixSparse>();
}

TEST_F(Test_IterativeRefinementBase, Test_NaN) {
    Test_NaN<MatrixDense>();
    Test_NaN<NoFillMatrixSparse>();
}
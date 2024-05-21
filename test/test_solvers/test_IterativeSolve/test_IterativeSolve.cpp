#include "../../test.h"

#include "test_IterativeSolve.h"

class TypedIterativeSolve_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestConstructors() {

        // Test with no initial guess and default parameters
        constexpr int n(6);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, n));
        Vector<T> soln(Vector<T>::Random(TestBase::bundle, 1));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        TypedIterativeSolveTestingMock<M, T> test_mock_no_guess(&typed_lin_sys, soln, default_args);

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess,
            Vector<double>::Ones(TestBase::bundle, n)
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.get_generic_soln(),
            Vector<double>::Ones(TestBase::bundle, n)
        );

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess_typed,
            Vector<double>::Ones(TestBase::bundle, n).template cast<T>()
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.typed_soln,
            Vector<double>::Ones(TestBase::bundle, n).template cast<T>()
        );

        EXPECT_EQ(test_mock_no_guess.max_iter, 100);
        EXPECT_EQ(test_mock_no_guess.target_rel_res, std::pow(10, -10));

        EXPECT_FALSE(test_mock_no_guess.initiated);
        EXPECT_FALSE(test_mock_no_guess.converged);
        EXPECT_FALSE(test_mock_no_guess.terminated);
        EXPECT_EQ(test_mock_no_guess.curr_iter, 0);
        EXPECT_EQ(test_mock_no_guess.res_norm_hist.size(), 1);
        EXPECT_NEAR(
            test_mock_no_guess.res_norm_hist[0],
            (b-A*test_mock_no_guess.init_guess).norm().get_scalar(),
            Tol<T>::gamma(n)
        );

        // Test with initial guess and explicit parameters
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        SolveArgPkg args;
        args.init_guess = init_guess;
        args.max_iter = n;
        args.target_rel_res = std::pow(10, -4);

        TypedIterativeSolveTestingMock<M, T> test_mock_guess(&typed_lin_sys, soln, args);

        ASSERT_VECTOR_EQ(test_mock_guess.init_guess, init_guess);
        ASSERT_VECTOR_EQ(
            test_mock_guess.generic_soln,
            init_guess.template cast<T>().template cast<double>()
        );

        ASSERT_VECTOR_EQ(test_mock_guess.init_guess_typed, init_guess.template cast<T>());
        ASSERT_VECTOR_EQ(test_mock_guess.typed_soln, init_guess.template cast<T>());

        EXPECT_EQ(test_mock_guess.max_iter, args.max_iter);
        EXPECT_EQ(test_mock_guess.target_rel_res, args.target_rel_res);

        EXPECT_FALSE(test_mock_guess.initiated);
        EXPECT_FALSE(test_mock_guess.converged);
        EXPECT_FALSE(test_mock_guess.terminated);
        EXPECT_EQ(test_mock_guess.curr_iter, 0);
        EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
        EXPECT_NEAR(
            test_mock_guess.res_norm_hist[0],
            (b - A*init_guess).norm().get_scalar(),
            Tol<T>::gamma(n)
        );

    }

    template <template <typename> typename M, typename T>
    void TestSolve() {

        constexpr int n(64);
        constexpr int max_iter(5);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_A.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_b.csv"))
        );

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        Vector<T> typed_soln(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_x.csv"))
        );
        Vector<double> init_guess(Vector<double>::Ones(TestBase::bundle, n));
        args.init_guess = init_guess;
        args.max_iter = max_iter;
        args.target_rel_res = (
            Tol<T>::roundoff() +
            ((b-A*typed_soln.template cast<double>()).norm()/(b-A*init_guess).norm()).get_scalar()
        );

        TypedIterativeSolveTestingMock<M, T> test_mock(&typed_lin_sys, typed_soln, args);

        EXPECT_NEAR(test_mock.get_relres(), 1., Tol<T>::gamma(n));
    
        test_mock.solve();

        // Check init_guess doesn't change
        ASSERT_VECTOR_EQ(test_mock.init_guess, init_guess);
        ASSERT_VECTOR_EQ(test_mock.init_guess_typed, init_guess.template cast<T>());

        // Check changed soln on iterate
        ASSERT_VECTOR_EQ(test_mock.typed_soln, typed_soln);

        // Check convergence
        EXPECT_TRUE(test_mock.initiated);
        EXPECT_TRUE(test_mock.converged);
        EXPECT_TRUE(test_mock.terminated);
        EXPECT_EQ(test_mock.curr_iter, 1);

        // Check residual history and relres correctly calculates
        EXPECT_EQ(test_mock.res_hist.cols(), max_iter+1);
        EXPECT_EQ(test_mock.res_hist.rows(), n);
        EXPECT_EQ(test_mock.res_norm_hist.size(), 2);
        ASSERT_VECTOR_NEAR(
            test_mock.res_hist.get_col(0).copy_to_vec(),
            b-A*init_guess,
            Tol<T>::gamma(n)
        );
        ASSERT_VECTOR_NEAR(
            test_mock.res_hist.get_col(1).copy_to_vec(),
            b-A*(typed_soln.template cast<double>()),
            Tol<T>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.res_norm_hist[0],
            (b-A*init_guess).norm().get_scalar(),
            Tol<T>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.res_norm_hist[1],
            (b-A*(typed_soln.template cast<double>())).norm().get_scalar(),
            Tol<T>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.get_relres(),
            ((b-A*(typed_soln.template cast<double>())).norm()/(b-A*init_guess).norm()).get_scalar(),
            Tol<T>::gamma(n)
        );

        if (*show_plots) { test_mock.view_relres_plot(); }

    }

    template <template <typename> typename M, typename T>
    void TestReset() {

        constexpr int n(64);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_A.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_b.csv"))
        );
        Vector<T> typed_soln(
            read_matrixCSV<Vector, T>(TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_x.csv"))
        );

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        TypedIterativeSolveTestingMock<M, T> test_mock(&typed_lin_sys, typed_soln, default_args);

        // Call solve and then reset
        test_mock.solve();
        test_mock.reset();

        // Check init_guess doesn't change
        ASSERT_VECTOR_EQ(test_mock.init_guess, Vector<double>::Ones(TestBase::bundle, n));

        // Check solve variables are all reset
        ASSERT_VECTOR_EQ(test_mock.typed_soln, Vector<T>::Ones(TestBase::bundle, n));
        EXPECT_FALSE(test_mock.initiated);
        EXPECT_FALSE(test_mock.converged);
        EXPECT_FALSE(test_mock.terminated);
        EXPECT_EQ(test_mock.curr_iter, 0);
        std::vector<double> init_res_norm_hist{(b - A*test_mock.init_guess).norm().get_scalar()};
        EXPECT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

    }

    template <template <typename> typename M>
    void TestMismatchedCols() {

        auto try_create_solve_mismatched_cols = []() {

            SolveArgPkg args;
            args.init_guess = Vector<double>::Ones(TestBase::bundle, 5, 1);

            GenericLinearSystem<M> gen_lin_sys(
                M<double>::Ones(TestBase::bundle, 64, 64),
                Vector<double>::Ones(TestBase::bundle, 64)
            );
            TypedLinearSystem<M, double> typed_lin_sys(&gen_lin_sys);

            TypedIterativeSolveTestingMock<M, double> test(
                &typed_lin_sys, Vector<double>::Ones(TestBase::bundle, 64), args
            );

        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_create_solve_mismatched_cols);

    }

    template <template <typename> typename M>
    void TestErrorNonSquare() {

        auto try_create_solve_non_square = [=]() {

            GenericLinearSystem<M> gen_lin_sys(
                M<double>::Ones(TestBase::bundle, 43, 64),
                Vector<double>::Ones(TestBase::bundle, 42)
            );
            TypedLinearSystem<M, double> typed_lin_sys(&gen_lin_sys);

            TypedIterativeSolveTestingMock<M, double> test(
                &typed_lin_sys, Vector<double>::Ones(TestBase::bundle, 64),  default_args
            );
        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_create_solve_non_square);

    }

};

TEST_F(TypedIterativeSolve_Test, TestConstructorsDouble_SOLVER) {
    TestConstructors<MatrixDense, double>();
    TestConstructors<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructorsSingle_SOLVER) {
    TestConstructors<MatrixDense, float>();
    TestConstructors<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructorsHalf_SOLVER) {
    TestConstructors<MatrixDense, half>();
    TestConstructors<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresHalf_SOLVER) {
    TestSolve<MatrixDense, half>();
    TestSolve<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresSingle_SOLVER) {
    TestSolve<MatrixDense, float>();
    TestSolve<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresDouble_SOLVER) {
    TestSolve<MatrixDense, double>();
    TestSolve<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestResetHalf_SOLVER) {
    TestReset<MatrixDense, half>();
    TestReset<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestResetSingle_SOLVER) {
    TestReset<MatrixDense, float>();
    TestReset<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestResetDouble_SOLVER) {
    TestReset<MatrixDense, double>();
    TestReset<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorMismatchedCols_SOLVER) {
    TestMismatchedCols<MatrixDense>();
    TestMismatchedCols<NoFillMatrixSparse>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorNonSquare_SOLVER) {
    TestErrorNonSquare<MatrixDense>();
    TestErrorNonSquare<NoFillMatrixSparse>();
}
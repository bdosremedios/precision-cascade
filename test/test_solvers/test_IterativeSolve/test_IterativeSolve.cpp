#include "../../test.h"

#include "test_IterativeSolve.h"

class TypedIterativeSolve_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestConstructors() {

        // Test with no initial guess and default parameters
        constexpr int n(6);
        M<double> A(M<double>::Random(*handle_ptr, n, n));
        Vector<double> b(Vector<double>::Random(*handle_ptr, n));
        Vector<T> soln(Vector<T>::Random(*handle_ptr, 1));
        TypedLinearSystem<M, T> typed_lin_sys(A, b);

        TypedIterativeSolveTestingMock<M, T> test_mock_no_guess(typed_lin_sys, soln, default_args);

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess,
            Vector<double>::Ones(*handle_ptr, n)
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.get_generic_soln(),
            Vector<double>::Ones(*handle_ptr, n)
        );

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess_typed,
            Vector<double>::Ones(*handle_ptr, n).template cast<T>()
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.typed_soln,
            Vector<double>::Ones(*handle_ptr, n).template cast<T>()
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
        Vector<double> init_guess(Vector<double>::Random(*handle_ptr, n));
        SolveArgPkg args;
        args.init_guess = init_guess;
        args.max_iter = n;
        args.target_rel_res = std::pow(10, -4);

        TypedIterativeSolveTestingMock<M, T> test_mock_guess(typed_lin_sys, soln, args);

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
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_A.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_b.csv"))
        );
        TypedLinearSystem<M, T> typed_lin_sys(A, b);

        SolveArgPkg args;
        Vector<T> typed_soln(
            read_matrixCSV<Vector, T>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_x.csv"))
        );
        Vector<double> init_guess(Vector<double>::Ones(*handle_ptr, n));
        args.init_guess = init_guess;
        args.max_iter = max_iter;
        args.target_rel_res = (
            Tol<T>::roundoff() +
            ((b-A*typed_soln.template cast<double>()).norm()/(b-A*init_guess).norm()).get_scalar()
        );

        TypedIterativeSolveTestingMock<M, T> test_mock(typed_lin_sys, typed_soln, args);

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
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_A.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_b.csv"))
        );
        TypedLinearSystem<M, T> typed_lin_sys(A, b);
        Vector<T> typed_soln(
            read_matrixCSV<Vector, T>(*handle_ptr, solve_matrix_dir / fs::path("conv_diff_64_x.csv"))
        );

        TypedIterativeSolveTestingMock<M, T> test_mock(typed_lin_sys, typed_soln, default_args);

        // Call solve and then reset
        test_mock.solve();
        test_mock.reset();

        // Check init_guess doesn't change
        ASSERT_VECTOR_EQ(test_mock.init_guess, Vector<double>::Ones(*handle_ptr, n));

        // Check solve variables are all reset
        ASSERT_VECTOR_EQ(test_mock.typed_soln, Vector<T>::Ones(*handle_ptr, n));
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
            args.init_guess = Vector<double>::Ones(*handle_ptr, 5, 1);
            TypedIterativeSolveTestingMock<M, double> test(
                TypedLinearSystem<M, double>(
                    M<double>::Ones(*handle_ptr, 64, 64),
                    Vector<double>::Ones(*handle_ptr, 64)
                ),
                Vector<double>::Ones(*handle_ptr, 5),
                args
            );
        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_create_solve_mismatched_cols);

    }

    template <template <typename> typename M>
    void TestErrorNonSquare() {

        auto try_create_solve_non_square = [=]() {
            TypedIterativeSolveTestingMock<M, double> test_mock(
                TypedLinearSystem<M, double>(
                    M<double>::Ones(*handle_ptr, 43, 64),
                    Vector<double>::Ones(*handle_ptr, 42)
                ),
                Vector<double>::Ones(*handle_ptr, 64),
                default_args
            );
        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_create_solve_non_square);

    }

};

TEST_F(TypedIterativeSolve_Test, TestConstructorsDouble) {
    TestConstructors<MatrixDense, double>();
    // TestConstructors<MatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructorsSingle) {
    TestConstructors<MatrixDense, float>();
    // TestConstructors<MatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructorsHalf) {
    TestConstructors<MatrixDense, half>();
    // TestConstructors<MatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresDouble) {
    TestSolve<MatrixDense, double>();
    // TestSolve<MatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresSingle) {
    TestSolve<MatrixDense, float>();
    // TestSolve<MatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelresHalf) {
    TestSolve<MatrixDense, half>();
    // TestSolve<MatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestResetDouble) {
    TestReset<MatrixDense, double>();
    // TestReset<MatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestResetSingle) {
    TestReset<MatrixDense, float>();
    // TestReset<MatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestResetHalf) {
    TestReset<MatrixDense, half>();
    // TestReset<MatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorMismatchedCols) {
    TestMismatchedCols<MatrixDense>();
    // TestMismatchedCols<MatrixSparse>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorNonSquare) {
    TestErrorNonSquare<MatrixDense>();
    // TestErrorNonSquare<MatrixSparse>();
}
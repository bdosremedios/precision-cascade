#include "../../test.h"

#include "test_IterativeSolve.h"

class TypedIterativeSolveTest: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestConstructors(double u) {

        // Test with no initial guess and default parameters
        constexpr int n(6);
        M<double> A = M<double>::Random(n, n);
        MatrixVector<double> b = MatrixVector<double>::Random(n);
        MatrixVector<T> soln = MatrixVector<T>::Random(1);
        TypedLinearSystem<M, T> typed_lin_sys(A, b);

        TypedIterativeSolveTestingMock<M, T> test_mock_no_guess(typed_lin_sys, soln, default_args);

        EXPECT_EQ(test_mock_no_guess.init_guess,
                  ((MatrixVector<double>::Ones(n).template cast<T>()).template cast<double>()));
        EXPECT_EQ(test_mock_no_guess.generic_soln,
                  ((MatrixVector<double>::Ones(n).template cast<T>()).template cast<double>()));

        EXPECT_EQ(test_mock_no_guess.init_guess_typed,
                  (MatrixVector<double>::Ones(n).template cast<T>()));
        EXPECT_EQ(test_mock_no_guess.typed_soln,
                  (MatrixVector<double>::Ones(n).template cast<T>()));

        EXPECT_EQ(test_mock_no_guess.max_iter, 100);
        EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

        EXPECT_FALSE(test_mock_no_guess.initiated);
        EXPECT_FALSE(test_mock_no_guess.converged);
        EXPECT_FALSE(test_mock_no_guess.terminated);
        EXPECT_EQ(test_mock_no_guess.curr_iter, 0);
        EXPECT_EQ(test_mock_no_guess.res_norm_hist.size(), 1);
        EXPECT_NEAR(test_mock_no_guess.res_norm_hist[0],
                    (b-A*MatrixVector<double>::Ones(n)).norm(),
                    gamma(n, u));

        // Test with initial guess and explicit parameters
        MatrixVector<double> init_guess = MatrixVector<double>::Random(n);
        SolveArgPkg args;
        args.init_guess = init_guess; args.max_iter = n; args.target_rel_res = pow(10, -4);
        TypedIterativeSolveTestingMock<M, T> test_mock_guess(typed_lin_sys, soln, args);

        EXPECT_EQ(test_mock_guess.init_guess, init_guess);
        EXPECT_EQ(test_mock_guess.generic_soln,
                  ((init_guess.template cast<T>()).template cast<double>()));

        EXPECT_EQ(test_mock_guess.init_guess_typed,
                  (init_guess.template cast<T>()));
        EXPECT_EQ(test_mock_guess.typed_soln,
                  (init_guess.template cast<T>()));

        EXPECT_EQ(test_mock_guess.max_iter, n);
        EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

        EXPECT_FALSE(test_mock_guess.initiated);
        EXPECT_FALSE(test_mock_guess.converged);
        EXPECT_FALSE(test_mock_guess.terminated);
        EXPECT_EQ(test_mock_guess.curr_iter, 0);
        EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
        EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*init_guess).norm(), gamma(n, u));

    }

    template <template <typename> typename M, typename T>
    void TestSolve(double u) {

        constexpr int n(64);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "conv_diff_64_A.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "conv_diff_64_b.csv");
        TypedLinearSystem<M, T> typed_lin_sys(A, b);

        MatrixVector<T> typed_soln = read_matrixCSV<MatrixVector, T>(solve_matrix_dir + "conv_diff_64_x.csv");
        MatrixVector<double> init_guess = MatrixVector<double>::Ones(n);

        SolveArgPkg args;
        args.init_guess = init_guess;
        args.target_rel_res = u + (b-A*typed_soln.template cast<double>()).norm()/(b-A*init_guess).norm();

        TypedIterativeSolveTestingMock<M, T> test_mock(typed_lin_sys, typed_soln, args);

        // Test start at 1 relres
        EXPECT_NEAR(test_mock.get_relres(), 1., gamma(n, u));

        // Call solve
        test_mock.solve();

        // Make sure init_guess doesn't change
        EXPECT_EQ(test_mock.init_guess, init_guess);
        EXPECT_EQ(test_mock.init_guess_typed, init_guess.template cast<T>());

        // Affirm changed soln on iterate
        EXPECT_EQ(test_mock.typed_soln, typed_soln);

        // Check convergence
        EXPECT_TRUE(test_mock.initiated);
        EXPECT_TRUE(test_mock.converged);
        EXPECT_TRUE(test_mock.terminated);
        EXPECT_EQ(test_mock.curr_iter, 1);

        // Check residual history and relres correctly calculates
        EXPECT_EQ(test_mock.res_hist.cols(), 2);
        EXPECT_EQ(test_mock.res_hist.rows(), n);
        EXPECT_EQ(test_mock.res_norm_hist.size(), 2);
        EXPECT_NEAR((test_mock.res_hist.col(0)-(b-A*init_guess)).norm(),
                    0.,
                    gamma(n, u));
        EXPECT_NEAR((test_mock.res_hist.col(1)-(b-A*(typed_soln.template cast<double>()))).norm(),
                    0.,
                    gamma(n, u));
        EXPECT_NEAR(test_mock.res_norm_hist[0],
                    (b-A*init_guess).norm(),
                    gamma(n, u));
        EXPECT_NEAR(test_mock.res_norm_hist[1],
                    (b-A*(typed_soln.template cast<double>())).norm(),
                    gamma(n, u));
        EXPECT_NEAR(test_mock.get_relres(),
                    (b-A*(typed_soln.template cast<double>())).norm()/(b-A*init_guess).norm(),
                    gamma(n, u));

        if (*show_plots) { test_mock.view_relres_plot(); }

    }

    template <template <typename> typename M, typename T>
    void TestReset() {

        constexpr int n(64);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "conv_diff_64_A.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "conv_diff_64_b.csv");
        TypedLinearSystem<M, T> typed_lin_sys(A, b);

        MatrixVector<T> typed_soln = read_matrixCSV<MatrixVector, T>(solve_matrix_dir + "conv_diff_64_x.csv");

        TypedIterativeSolveTestingMock<M, T> test_mock(typed_lin_sys, typed_soln, default_args);

        // Call solve and then reset
        test_mock.solve();
        test_mock.reset();

        // Make sure init_guess doesn't change
        EXPECT_EQ(test_mock.init_guess, (MatrixVector<double>::Ones(n)));

        // Check solve variables are all reset
        EXPECT_EQ(test_mock.typed_soln, (MatrixVector<T>::Ones(n)));
        EXPECT_FALSE(test_mock.initiated);
        EXPECT_FALSE(test_mock.converged);
        EXPECT_FALSE(test_mock.terminated);
        EXPECT_EQ(test_mock.curr_iter, 0);
        vector<double> init_res_norm_hist = {(b - A*MatrixVector<double>::Ones(n)).norm()};
        EXPECT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

    }

    template <template <typename> typename M>
    void TestMismatchedCols() {

        try {
            SolveArgPkg args; args.init_guess = MatrixVector<double>::Ones(5, 1);
            TypedIterativeSolveTestingMock<M, double> test(
                TypedLinearSystem<M, double>(M<double>::Ones(64, 64),
                                             MatrixVector<double>::Ones(64)),
                MatrixVector<double>::Ones(5),
                args
            );
            FAIL();
        } catch (runtime_error e) { cout << e.what() << endl; }

    }

    template <template <typename> typename M>
    void TestErrorNonSquare() {

        try {
            TypedIterativeSolveTestingMock<M, double> test_mock(
                TypedLinearSystem<M, double>(M<double>::Ones(43, 64),
                                            MatrixVector<double>::Ones(42)),
                MatrixVector<double>::Ones(64),
                default_args
            );
            FAIL();
        } catch (runtime_error e) { cout << e.what() << endl; }

    }

};

TEST_F(TypedIterativeSolveTest, TestConstructorsDouble_Dense) {
    TestConstructors<MatrixDense, double>(u_dbl);
}
TEST_F(TypedIterativeSolveTest, TestConstructorsDouble_Sparse) {
    TestConstructors<MatrixSparse, double>(u_dbl);
}

TEST_F(TypedIterativeSolveTest, TestConstructorsSingle_Dense) {
    TestConstructors<MatrixDense, float>(u_sgl);
}
TEST_F(TypedIterativeSolveTest, TestConstructorsSingle_Sparse) {
    TestConstructors<MatrixSparse, float>(u_sgl);
}

TEST_F(TypedIterativeSolveTest, TestConstructorsHalf_Dense) {
    TestConstructors<MatrixDense, half>(u_hlf);
}
TEST_F(TypedIterativeSolveTest, TestConstructorsHalf_Sparse) {
    TestConstructors<MatrixSparse, half>(u_hlf);
}

TEST_F(TypedIterativeSolveTest, TestSolveAndRelresDouble_Dense) { TestSolve<MatrixDense, double>(u_dbl); }
TEST_F(TypedIterativeSolveTest, TestSolveAndRelresDouble_Sparse) { TestSolve<MatrixSparse, double>(u_dbl); }

TEST_F(TypedIterativeSolveTest, TestSolveAndRelresSingle_Dense) { TestSolve<MatrixDense, float>(u_sgl); }
TEST_F(TypedIterativeSolveTest, TestSolveAndRelresSingle_Sparse) { TestSolve<MatrixSparse, float>(u_sgl); }

TEST_F(TypedIterativeSolveTest, TestSolveAndRelresHalf_Dense) { TestSolve<MatrixDense, half>(u_hlf); }
TEST_F(TypedIterativeSolveTest, TestSolveAndRelresHalf_Sparse) { TestSolve<MatrixSparse, half>(u_hlf); }

TEST_F(TypedIterativeSolveTest, TestResetDouble_Dense) { TestReset<MatrixDense, double>(); }
TEST_F(TypedIterativeSolveTest, TestResetDouble_Sparse) { TestReset<MatrixSparse, double>(); }

TEST_F(TypedIterativeSolveTest, TestResetSingle_Dense) { TestReset<MatrixDense, float>(); }
TEST_F(TypedIterativeSolveTest, TestResetSingle_Sparse) { TestReset<MatrixSparse, float>(); }

TEST_F(TypedIterativeSolveTest, TestResetHalf_Dense) { TestReset<MatrixDense, half>(); }
TEST_F(TypedIterativeSolveTest, TestResetHalf_Sparse) { TestReset<MatrixSparse, half>(); }

TEST_F(TypedIterativeSolveTest, TestErrorMismatchedCols_Dense) { TestMismatchedCols<MatrixDense>(); }
TEST_F(TypedIterativeSolveTest, TestErrorMismatchedCols_Sparse) { TestMismatchedCols<MatrixSparse>(); }

TEST_F(TypedIterativeSolveTest, TestErrorNonSquare_Dense) { TestErrorNonSquare<MatrixDense>(); }
TEST_F(TypedIterativeSolveTest, TestErrorNonSquare_Sparse) { TestErrorNonSquare<MatrixSparse>(); }
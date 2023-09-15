#include "../../test.h"

#include "test_IterativeSolve.h"

class TypedIterativeSolveTest: public TestBase {

    public:

    template <template<typename> typename M, typename T>
    void TestConstructors(double u) {

        // Test with no initial guess and default parameters
        constexpr int n(6);
        M<double> A = M<double>::Random(n, n);
        MatrixVector<double> b = MatrixVector<double>::Random(n);
        MatrixVector<T> soln = MatrixVector<T>::Random(1);
        TypedIterativeSolveTestingMock<M, T> test_mock_no_guess(A, b, soln, default_args);

        EXPECT_EQ(test_mock_no_guess.m, n);
        EXPECT_EQ(test_mock_no_guess.n, n);

        EXPECT_EQ(test_mock_no_guess.A, A);
        EXPECT_EQ(test_mock_no_guess.b, b);
        EXPECT_EQ(test_mock_no_guess.init_guess,
                    ((MatrixVector<double>::Ones(n).template cast<T>()).template cast<double>()));
        EXPECT_EQ(test_mock_no_guess.generic_soln,
                    ((MatrixVector<double>::Ones(n).template cast<T>()).template cast<double>()));

        EXPECT_EQ(test_mock_no_guess.A_T, A.template cast<T>());
        EXPECT_EQ(test_mock_no_guess.b_T, b.template cast<T>());
        EXPECT_EQ(test_mock_no_guess.init_guess_T,
                    (MatrixVector<double>::Ones(n).template cast<T>()));
        EXPECT_EQ(test_mock_no_guess.typed_soln,
                    (MatrixVector<double>::Ones(n).template cast<T>()));

        EXPECT_EQ(test_mock_no_guess.max_iter, 100);
        EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

        EXPECT_FALSE(test_mock_no_guess.initiated);
        EXPECT_FALSE(test_mock_no_guess.converged);
        EXPECT_FALSE(test_mock_no_guess.terminated);
        EXPECT_EQ(test_mock_no_guess.curr_iter, 0);
        vector<double> init_res_norm_hist{(b - A*MatrixVector<double>::Ones(n)).norm()};
        EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

        // Test with initial guess and explicit parameters
        MatrixVector<double> init_guess = MatrixVector<double>::Random(n);
        SolveArgPkg args;
        args.init_guess = init_guess; args.max_iter = n; args.target_rel_res = pow(10, -4);
        TypedIterativeSolveTestingMock<M, T> test_mock_guess(A, b, soln, args);

        EXPECT_EQ(test_mock_guess.A, A);
        EXPECT_EQ(test_mock_guess.b, b);
        EXPECT_EQ(test_mock_guess.init_guess, init_guess);
        EXPECT_EQ(test_mock_guess.generic_soln,
                    ((init_guess.template cast<T>()).template cast<double>()));

        EXPECT_EQ(test_mock_guess.A_T, (A.template cast<T>()));
        EXPECT_EQ(test_mock_guess.b_T, (b.template cast<T>()));
        EXPECT_EQ(test_mock_guess.init_guess_T,
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

    
    template <template<typename> typename M, typename T>
    void TestSolve() {

        constexpr int n(64);
        MatrixDense<double> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv");
        MatrixVector<double> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv");
        MatrixVector<double> typed_soln = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv");
        MatrixVector<double> init_guess = MatrixVector<double>::Ones(n);
        TypedIterativeSolveTestingMock<MatrixDense, double> test_mock(A, b, typed_soln, default_args);

        // Test start at 1 relres
        EXPECT_NEAR(test_mock.get_relres(), 1., gamma(n, u_dbl));

        // Call solve
        test_mock.solve();

        // Make sure other variables don't change
        EXPECT_EQ(test_mock.A, A);
        EXPECT_EQ(test_mock.b, b);
        EXPECT_EQ(test_mock.m, n);
        EXPECT_EQ(test_mock.n, n);
        EXPECT_EQ(test_mock.init_guess, init_guess);
        EXPECT_EQ(test_mock.typed_soln, typed_soln);

        // Check convergence
        EXPECT_TRUE(test_mock.initiated);
        EXPECT_TRUE(test_mock.converged);
        EXPECT_TRUE(test_mock.terminated);
        EXPECT_EQ(test_mock.curr_iter, 1);

        // Check residual history matches size and has initial norm and solved norm
        EXPECT_EQ(test_mock.res_hist.cols(), 2);
        EXPECT_EQ(test_mock.res_hist.rows(), n);
        EXPECT_EQ(test_mock.res_norm_hist.size(), 2);
        EXPECT_EQ(test_mock.res_hist.col(0), b-A*init_guess);
        EXPECT_EQ(test_mock.res_hist.col(1), b-A*typed_soln);
        EXPECT_EQ(test_mock.res_norm_hist[0], (b-A*init_guess).norm());
        EXPECT_EQ(test_mock.res_norm_hist[1], (b-A*typed_soln).norm());

        // Test start end at (b-A*typed_soln).norm() relres with right solution
        EXPECT_NEAR(test_mock.get_relres(), (b-A*typed_soln).norm()/(b-A*init_guess).norm(), gamma(n, u_dbl));

        if (*show_plots) { test_mock.view_relres_plot(); }

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

TEST_F(TypedIterativeSolveTest, TestSolveAndRelres) {

}

TEST_F(TypedIterativeSolveTest, TestReset) {

    constexpr int n(64);
    MatrixDense<double> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    MatrixVector<double> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    MatrixVector<double> typed_soln(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    TypedIterativeSolveTestingMock<MatrixDense, double> test_mock(A, b, typed_soln, default_args);

    // Call solve and then reset
    test_mock.solve();
    test_mock.reset();

    // Make sure other variables don't change
    EXPECT_EQ(test_mock.A, A);
    EXPECT_EQ(test_mock.b, b);
    EXPECT_EQ(test_mock.m, n);
    EXPECT_EQ(test_mock.n, n);
    EXPECT_EQ(test_mock.init_guess, (MatrixVector<double>::Ones(n)));

    // Check solve variables are all reset
    EXPECT_EQ(test_mock.typed_soln, (MatrixVector<double>::Ones(n)));
    EXPECT_FALSE(test_mock.initiated);
    EXPECT_FALSE(test_mock.converged);
    EXPECT_FALSE(test_mock.terminated);
    EXPECT_EQ(test_mock.curr_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*MatrixVector<double>::Ones(n)).norm()};
    EXPECT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

}

TEST_F(TypedIterativeSolveTest, TestErrorEmptyMatrix) {

    try {
        TypedIterativeSolveTestingMock<MatrixDense, double> test(
            MatrixDense<double>::Ones(0, 0),
            MatrixVector<double>::Ones(0),
            MatrixVector<double>::Ones(0),
            default_args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(TypedIterativeSolveTest, TestErrorMismatchedCols) {

    try {
        SolveArgPkg args; args.init_guess = MatrixDense<double>::Ones(5, 1);
        TypedIterativeSolveTestingMock<MatrixDense, double> test(
            MatrixDense<double>::Ones(64, 64),
            MatrixVector<double>::Ones(64),
            MatrixVector<double>::Ones(5),
            args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(TypedIterativeSolveTest, TestErrorMismatchedRows) {

    try {
        TypedIterativeSolveTestingMock<MatrixDense, double> test(
            MatrixDense<double>::Ones(64, 64),
            MatrixVector<double>::Ones(8),
            MatrixVector<double>::Ones(64),
            default_args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(TypedIterativeSolveTest, TestErrorNonSquare) {

    try {
        TypedIterativeSolveTestingMock<MatrixDense, double> test_mock(
            MatrixDense<double>::Ones(43, 64),
            MatrixVector<double>::Ones(43),
            MatrixVector<double>::Ones(64),
            default_args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }
 
}
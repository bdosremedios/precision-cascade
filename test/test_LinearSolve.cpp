#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/LinearSolve.h"

#include <cmath>
#include <vector>
#include <iostream>

using Eigen::Matrix, Eigen::Dynamic, Eigen::half;

using read_matrix::read_matrix_csv;

using std::pow;
using std::vector;
using std::cout, std::endl;

class LinearSolveTest: public TestBase {};

TEST_F(LinearSolveTest, TestConstructorsDouble) {

    // Test with no initial guess and default parameters
    constexpr int n(6);
    Matrix<double, Dynamic, Dynamic> A(Matrix<double, n, n>::Random());
    Matrix<double, Dynamic, 1> b(Matrix<double, n, 1>::Random());
    Matrix<double, Dynamic, 1> soln(Matrix<double, 1, 1>::Ones());
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<double, n, 1>::Ones()));
    EXPECT_EQ(test_mock_no_guess.m, n); EXPECT_EQ(test_mock_no_guess.n, n);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<double, n, 1>::Ones()));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist{(b - A*Matrix<double, n, 1>::Ones()).norm()};
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<double, Dynamic, 1> x_0(Matrix<double, n, 1>::Random());
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, n, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, n); EXPECT_EQ(test_mock_guess.n, n);
    EXPECT_EQ(test_mock_guess.max_outer_iter, n);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(n, u_dbl));

}

TEST_F(LinearSolveTest, TestConstructorsSingle) {

    constexpr int n(6);
    // Test with no initial guess and default parameters
    Matrix<float, Dynamic, Dynamic> A(Matrix<float, n, n>::Random());
    Matrix<float, Dynamic, 1> b(Matrix<float, n, 1>::Random());
    Matrix<float, Dynamic, 1> soln(Matrix<float, 1, 1>::Ones());
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<float, n, 1>::Ones()));
    EXPECT_EQ(test_mock_no_guess.m, n); EXPECT_EQ(test_mock_no_guess.n, n);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<float, n, 1>::Ones()));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist{static_cast<double>((b - A*Matrix<float, n, 1>::Ones()).norm())};
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<float, Dynamic, 1> x_0(Matrix<float, n, 1>::Random());
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, n, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, n); EXPECT_EQ(test_mock_guess.n, n);
    EXPECT_EQ(test_mock_guess.max_outer_iter, n);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(n, u_sgl));

}

TEST_F(LinearSolveTest, TestConstructorsHalf) {

    // Test with no initial guess and default parameters
    constexpr int n(6);
    Matrix<half, Dynamic, Dynamic> A(Matrix<half, n, n>::Random());
    Matrix<half, Dynamic, 1> b(Matrix<half, n, 1>::Random());
    Matrix<half, Dynamic, 1> soln(Matrix<half, n, 1>::Ones());
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<half, n, 1>::Ones()));
    EXPECT_EQ(test_mock_no_guess.m, n); EXPECT_EQ(test_mock_no_guess.n, n);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<half, n, 1>::Ones()));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {static_cast<double>((b - A*Matrix<half, n, 1>::Ones()).norm())};
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<half, Dynamic, 1> x_0(Matrix<half, n, 1>::Random());
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, n, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, n); EXPECT_EQ(test_mock_guess.n, n);
    EXPECT_EQ(test_mock_guess.max_outer_iter, n);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(n, u_hlf));

}

TEST_F(LinearSolveTest, TestSolveAndRelres) {

    constexpr int n(64);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    Matrix<double, Dynamic, 1> x_0(Matrix<double, n, 1>::Ones());
    LinearSolveTestingMock test_mock(A, b, x);

    // Test start at 1 relres
    EXPECT_NEAR(test_mock.get_relres(), 1., gamma(n, u_dbl));

    // Call solve
    test_mock.solve();

    // Make sure other variables don't change
    EXPECT_EQ(test_mock.A, A);
    EXPECT_EQ(test_mock.b, b);
    EXPECT_EQ(test_mock.m, n);
    EXPECT_EQ(test_mock.n, n);
    EXPECT_EQ(test_mock.x_0, x_0);
    EXPECT_EQ(test_mock.x, x);

    // Check convergence
    EXPECT_TRUE(test_mock.initiated);
    EXPECT_TRUE(test_mock.converged);
    EXPECT_TRUE(test_mock.terminated);
    EXPECT_EQ(test_mock.curr_outer_iter, 1);

    // Check residual history matches size and has initial norm and solved norm
    EXPECT_EQ(test_mock.res_hist.cols(), 2);
    EXPECT_EQ(test_mock.res_hist.rows(), n);
    EXPECT_EQ(test_mock.res_norm_hist.size(), 2);
    EXPECT_EQ(test_mock.res_hist.col(0), b-A*x_0);
    EXPECT_EQ(test_mock.res_hist.col(1), b-A*x);
    EXPECT_EQ(test_mock.res_norm_hist[0], (b-A*x_0).norm());
    EXPECT_EQ(test_mock.res_norm_hist[1], (b-A*x).norm());

    // Test start end at (b-A*x).norm() relres with right solution
    EXPECT_NEAR(test_mock.get_relres(), (b-A*x).norm()/(b-A*x_0).norm(), gamma(n, u_dbl));

    if (show_plots) { test_mock.view_relres_plot(); }

}

TEST_F(LinearSolveTest, TestReset) {

    constexpr int n(64);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    LinearSolveTestingMock test_mock(A, b, x);

    // Call solve and then reset
    test_mock.solve();
    test_mock.reset();

    // Make sure other variables don't change
    EXPECT_EQ(test_mock.A, A);
    EXPECT_EQ(test_mock.b, b);
    EXPECT_EQ(test_mock.m, n);
    EXPECT_EQ(test_mock.n, n);
    EXPECT_EQ(test_mock.x_0, (Matrix<double, n, 1>::Ones()));

    // Check solve variables are all reset
    EXPECT_EQ(test_mock.x, (Matrix<double, n, 1>::Ones()));
    EXPECT_FALSE(test_mock.initiated);
    EXPECT_FALSE(test_mock.converged);
    EXPECT_FALSE(test_mock.terminated);
    EXPECT_EQ(test_mock.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*Matrix<double, n, 1>::Ones()).norm()};
    EXPECT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

}

TEST_F(LinearSolveTest, TestErrorEmptyMatrix) {

    try {
        LinearSolveTestingMock<double> test(
            Matrix<double, Dynamic, Dynamic>::Ones(0, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(0, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(0, 1));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(LinearSolveTest, TestErrorMismatchedCols) {

    try {
        LinearSolveTestingMock<double> test(
            Matrix<double, Dynamic, Dynamic>::Ones(64, 64),
            Matrix<double, Dynamic, Dynamic>::Ones(64, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(5, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(5, 1));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(LinearSolveTest, TestErrorMismatchedRows) {

    try {
        LinearSolveTestingMock<double> test(
            Matrix<double, Dynamic, Dynamic>::Ones(64, 64),
            Matrix<double, Dynamic, Dynamic>::Ones(8, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(64, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(64, 1));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(LinearSolveTest, TestErrorNonSquare) {

    try {
        LinearSolveTestingMock<double> test_mock(
            Matrix<double, Dynamic, Dynamic>::Ones(43, 64),
            Matrix<double, Dynamic, Dynamic>::Ones(43, 1),
            Matrix<double, Dynamic, Dynamic>::Ones(64, 1));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }
 
}
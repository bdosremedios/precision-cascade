#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/LinearSolve.h"

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::half;

using read_matrix::read_matrix_csv;

using std::pow;
using std::string;
using std::vector;
using std::cout, std::endl;

class LinearSolveTest: public TestBase {};

TEST_F(LinearSolveTest, TestConstructorsDouble) {

    // Test with no initial guess and default parameters
    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(6, 6);
    Matrix<double, Dynamic, 1> b = Matrix<double, Dynamic, Dynamic>::Random(6, 1);
    Matrix<double, Dynamic, 1> soln = Matrix<double, Dynamic, Dynamic>::Ones(1, 1);
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<double, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_EQ(test_mock_no_guess.m, 6); EXPECT_EQ(test_mock_no_guess.n, 6);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<double, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*Matrix<double, Dynamic, Dynamic>::Ones(6, 1)).norm()};
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<double, Dynamic, 1> x_0 = Matrix<double, Dynamic, Dynamic>::Random(6, 1);
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, 6, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, 6); EXPECT_EQ(test_mock_guess.n, 6);
    EXPECT_EQ(test_mock_guess.max_outer_iter, 6);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(6, u_dbl));

}

TEST_F(LinearSolveTest, TestConstructorsSingle) {

    // Test with no initial guess and default parameters
    Matrix<float, Dynamic, Dynamic> A = Matrix<float, Dynamic, Dynamic>::Random(6, 6);
    Matrix<float, Dynamic, 1> b = Matrix<float, Dynamic, Dynamic>::Random(6, 1);
    Matrix<float, Dynamic, 1> soln = Matrix<float, Dynamic, Dynamic>::Ones(1, 1);
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<float, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_EQ(test_mock_no_guess.m, 6); EXPECT_EQ(test_mock_no_guess.n, 6);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<float, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {
        static_cast<double>((b - A*Matrix<float, Dynamic, Dynamic>::Ones(6, 1)).norm())
    };
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<float, Dynamic, 1> x_0 = Matrix<float, Dynamic, Dynamic>::Random(6, 1);
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, 6, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, 6); EXPECT_EQ(test_mock_guess.n, 6);
    EXPECT_EQ(test_mock_guess.max_outer_iter, 6);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(6, u_sgl));

}

TEST_F(LinearSolveTest, TestConstructorsHalf) {

    // Test with no initial guess and default parameters
    Matrix<half, Dynamic, Dynamic> A = Matrix<half, Dynamic, Dynamic>::Random(6, 6);
    Matrix<half, Dynamic, 1> b = Matrix<half, Dynamic, Dynamic>::Random(6, 1);
    Matrix<half, Dynamic, 1> soln = Matrix<half, Dynamic, Dynamic>::Ones(1, 1);
    LinearSolveTestingMock test_mock_no_guess(A, b, soln);

    EXPECT_EQ(test_mock_no_guess.A, A); EXPECT_EQ(test_mock_no_guess.b, b);
    EXPECT_EQ(test_mock_no_guess.x_0, (Matrix<half, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_EQ(test_mock_no_guess.m, 6); EXPECT_EQ(test_mock_no_guess.n, 6);
    EXPECT_EQ(test_mock_no_guess.max_outer_iter, 100);
    EXPECT_EQ(test_mock_no_guess.target_rel_res, pow(10, -10));

    EXPECT_EQ(test_mock_no_guess.x, (Matrix<half, Dynamic, Dynamic>::Ones(6, 1)));
    EXPECT_FALSE(test_mock_no_guess.initiated);
    EXPECT_FALSE(test_mock_no_guess.converged);
    EXPECT_FALSE(test_mock_no_guess.terminated);
    EXPECT_EQ(test_mock_no_guess.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {
        static_cast<double>((b - A*Matrix<half, Dynamic, Dynamic>::Ones(6, 1)).norm())
    };
    EXPECT_EQ(test_mock_no_guess.res_norm_hist, init_res_norm_hist);

    // Test with initial guess and explicit parameters
    Matrix<half, Dynamic, 1> x_0 = Matrix<half, Dynamic, Dynamic>::Random(6, 1);
    LinearSolveTestingMock test_mock_guess(A, b, x_0, soln, 6, pow(10, -4));

    EXPECT_EQ(test_mock_guess.A, A); EXPECT_EQ(test_mock_guess.b, b);
    EXPECT_EQ(test_mock_guess.x_0, x_0);
    EXPECT_EQ(test_mock_guess.m, 6); EXPECT_EQ(test_mock_guess.n, 6);
    EXPECT_EQ(test_mock_guess.max_outer_iter, 6);
    EXPECT_EQ(test_mock_guess.target_rel_res, pow(10, -4));

    EXPECT_EQ(test_mock_guess.x, x_0);
    EXPECT_FALSE(test_mock_guess.initiated);
    EXPECT_FALSE(test_mock_guess.converged);
    EXPECT_FALSE(test_mock_guess.terminated);
    EXPECT_EQ(test_mock_guess.curr_outer_iter, 0);
    EXPECT_EQ(test_mock_guess.res_norm_hist.size(), 1);
    EXPECT_NEAR(test_mock_guess.res_norm_hist[0], (b - A*x_0).norm(), gamma(6, u_hlf));

}

TEST_F(LinearSolveTest, TestSolveAndRelres) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    Matrix<double, Dynamic, 1> x_0 = Matrix<double, Dynamic, Dynamic>::Ones(64, 1);
    LinearSolveTestingMock test_mock(A, b, x);

    // Test start at 1 relres
    EXPECT_NEAR(test_mock.get_relres(), 1., gamma(64, u_dbl));

    // Call solve
    test_mock.solve();

    // Make sure other variables don't change
    EXPECT_EQ(test_mock.A, A);
    EXPECT_EQ(test_mock.b, b);
    EXPECT_EQ(test_mock.m, 64);
    EXPECT_EQ(test_mock.n, 64);
    EXPECT_EQ(test_mock.x_0, x_0);
    EXPECT_EQ(test_mock.x, x);

    // Check convergence
    EXPECT_TRUE(test_mock.initiated);
    EXPECT_TRUE(test_mock.converged);
    EXPECT_TRUE(test_mock.terminated);
    EXPECT_EQ(test_mock.curr_outer_iter, 1);

    // Check residual history matches size and has initial norm and solved norm
    EXPECT_EQ(test_mock.res_hist.cols(), 2);
    EXPECT_EQ(test_mock.res_hist.rows(), 64);
    EXPECT_EQ(test_mock.res_norm_hist.size(), 2);
    EXPECT_EQ(test_mock.res_hist.col(0), b-A*x_0);
    EXPECT_EQ(test_mock.res_hist.col(1), b-A*x);
    EXPECT_EQ(test_mock.res_norm_hist[0], (b-A*x_0).norm());
    EXPECT_EQ(test_mock.res_norm_hist[1], (b-A*x).norm());

    // Test start end at (b-A*x).norm() relres with right solution
    EXPECT_NEAR(test_mock.get_relres(), (b-A*x).norm()/(b-A*x_0).norm(), gamma(64, u_dbl));

    test_mock.view_relres_plot();

}

TEST_F(LinearSolveTest, TestReset) {
    
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
    EXPECT_EQ(test_mock.m, 64);
    EXPECT_EQ(test_mock.n, 64);
    EXPECT_EQ(test_mock.x_0, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));

    // Check solve variables are all reset
    EXPECT_EQ(test_mock.x, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));
    EXPECT_FALSE(test_mock.initiated);
    EXPECT_FALSE(test_mock.converged);
    EXPECT_FALSE(test_mock.terminated);
    EXPECT_EQ(test_mock.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*Matrix<double, Dynamic, Dynamic>::Ones(64, 1)).norm()};
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
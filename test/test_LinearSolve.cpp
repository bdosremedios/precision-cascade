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
using read_matrix::read_matrix_csv;

using std::pow;
using std::string;
using std::vector;
using std::cout, std::endl;

class LinearSolveTest: public TestBase {};

TEST_F(LinearSolveTest, TestConstructor64) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    LinearSolveTestingMock test_mock(A, b, x, 64, pow(10, -12));

    ASSERT_EQ(test_mock.A, A);
    ASSERT_EQ(test_mock.b, b);
    ASSERT_EQ(test_mock.m, 64);
    ASSERT_EQ(test_mock.n, 64);
    ASSERT_EQ(test_mock.max_outer_iter, 64);
    ASSERT_EQ(test_mock.target_rel_res, pow(10, -12));
    ASSERT_EQ(test_mock.x_0, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));

    ASSERT_EQ(test_mock.x, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));
    ASSERT_FALSE(test_mock.initiated);
    ASSERT_FALSE(test_mock.converged);
    ASSERT_FALSE(test_mock.terminated);
    ASSERT_EQ(test_mock.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*Matrix<double, Dynamic, Dynamic>::Ones(64, 1)).norm()};
    ASSERT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

}

TEST_F(LinearSolveTest, TestConstructor256) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_x.csv"));
    Matrix<double, Dynamic, 1> x_0(Matrix<double, Dynamic, Dynamic>::Random(256, 1));
    LinearSolveTestingMock test_mock(A, b, x_0, x, 256, pow(10, -12));

    ASSERT_EQ(test_mock.A, A);
    ASSERT_EQ(test_mock.b, b);
    ASSERT_EQ(test_mock.m, 256);
    ASSERT_EQ(test_mock.n, 256);
    ASSERT_EQ(test_mock.max_outer_iter, 256);
    ASSERT_EQ(test_mock.target_rel_res, pow(10, -12));
    ASSERT_EQ(test_mock.x_0, x_0);

    ASSERT_EQ(test_mock.x, x_0);
    ASSERT_FALSE(test_mock.initiated);
    ASSERT_FALSE(test_mock.converged);
    ASSERT_FALSE(test_mock.terminated);
    ASSERT_EQ(test_mock.curr_outer_iter, 0);
    ASSERT_EQ(test_mock.res_norm_hist.size(), 1);
    ASSERT_NEAR(test_mock.res_norm_hist[0], (b - A*x_0).norm(), gamma(256, u_dbl));

}

TEST_F(LinearSolveTest, TestSolve) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    Matrix<double, Dynamic, 1> x(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    LinearSolveTestingMock test_mock(A, b, x);

    // Call solve
    test_mock.solve();

    // Make sure other variables don't change
    ASSERT_EQ(test_mock.A, A);
    ASSERT_EQ(test_mock.b, b);
    ASSERT_EQ(test_mock.m, 64);
    ASSERT_EQ(test_mock.n, 64);
    ASSERT_EQ(test_mock.x_0, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));
    ASSERT_EQ(test_mock.x, x);

    // Check convergence
    ASSERT_TRUE(test_mock.initiated);
    ASSERT_TRUE(test_mock.converged);
    ASSERT_TRUE(test_mock.terminated);
    ASSERT_EQ(test_mock.curr_outer_iter, 1);

    // Check residual history matches size and has initial norm and solved norm
    ASSERT_EQ(test_mock.res_hist.cols(), 2);
    ASSERT_EQ(test_mock.res_hist.rows(), 64);
    ASSERT_EQ(test_mock.res_norm_hist.size(), 2);
    ASSERT_EQ(test_mock.res_hist.col(0), b-A*test_mock.x_0);
    ASSERT_EQ(test_mock.res_hist.col(1), b-A*x);
    ASSERT_EQ(test_mock.res_norm_hist[0], (b-A*test_mock.x_0).norm());
    ASSERT_EQ(test_mock.res_norm_hist[1], (b-A*x).norm());

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
    ASSERT_EQ(test_mock.A, A);
    ASSERT_EQ(test_mock.b, b);
    ASSERT_EQ(test_mock.m, 64);
    ASSERT_EQ(test_mock.n, 64);
    ASSERT_EQ(test_mock.x_0, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));

    // Check solve variables are all reset
    ASSERT_EQ(test_mock.x, (Matrix<double, Dynamic, Dynamic>::Ones(64, 1)));
    ASSERT_FALSE(test_mock.initiated);
    ASSERT_FALSE(test_mock.converged);
    ASSERT_FALSE(test_mock.terminated);
    ASSERT_EQ(test_mock.curr_outer_iter, 0);
    vector<double> init_res_norm_hist = {(b - A*Matrix<double, Dynamic, Dynamic>::Ones(64, 1)).norm()};
    ASSERT_EQ(test_mock.res_norm_hist, init_res_norm_hist);

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
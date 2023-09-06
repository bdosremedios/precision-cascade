#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include "../test.h"

#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;

using read_matrix::read_matrix_csv;

using std::cout, std::endl;

class GMRESSingleSolveTest: public TestBase {};

TEST_F(GMRESSingleSolveTest, SolveConvDiff64) {

    constexpr int n(64);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleSolveTest, SolveConvDiff256) {

    constexpr int n(256);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleSolveTest, SolveConvDiff1024_LONGRUNTIME) {

    constexpr int n(1024);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleSolveTest, SolveRand20) {

    constexpr int n(20);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleSolveTest, Solve3Eigs) {

    constexpr int n(25);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_EQ(gmres_solve_s.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleSolveTest, DivergeBeyondSingleCapabilities) {

    constexpr int n(64);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    // Check convergence under single capabilities
    GMRESSolve<float> gmres_solve_s(A, b, u_sgl, n, conv_tol_sgl);

    gmres_solve_s.solve();
    if (show_plots) { gmres_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

    // Check divergence beyond single capability of the single machine epsilon
    GMRESSolve<float> gmres_solve_s_to_fail(A, b, u_sgl, n, 0.1*u_sgl);
    gmres_solve_s_to_fail.solve();
    if (show_plots) { gmres_solve_s_to_fail.view_relres_plot("log"); }
    
    EXPECT_FALSE(gmres_solve_s_to_fail.check_converged());
    EXPECT_GT(gmres_solve_s_to_fail.get_relres(), 0.1*u_sgl);

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;

using read_matrix::read_matrix_csv;

using std::cout, std::endl;

class GMRESDoubleSolveTest: public TestBase {};

TEST_F(GMRESDoubleSolveTest, SolveConvDiff64) {

    constexpr int n(64);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolve<double> gmres_solve_d(A, b, u_dbl, n, conv_tol_dbl);

    gmres_solve_d.solve();
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv"));
    EXPECT_LE((gmres_solve_d.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff256) {

    constexpr int n(256);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolve<double> gmres_solve_d(A, b, u_dbl, n, conv_tol_dbl);

    gmres_solve_d.solve();
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_x.csv"));
    EXPECT_LE((gmres_solve_d.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff1024_LONGRUNTIME) {

    constexpr int n(1024);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolve<double> gmres_solve_d(A, b, u_dbl, n, conv_tol_dbl);

    gmres_solve_d.solve();
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_x.csv"));
    EXPECT_LE((gmres_solve_d.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleSolveTest, SolveRand20) {

    constexpr int n(20);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolve<double> gmres_solve_d(A, b, u_dbl, n, conv_tol_dbl);

    gmres_solve_d.solve();
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_20_rand.csv"));
    EXPECT_LE((gmres_solve_d.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleSolveTest, Solve3Eigs) {

    constexpr int n(25);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolve<double> gmres_solve_d(A, b, u_dbl, n, conv_tol_dbl);

    gmres_solve_d.solve();
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_EQ(gmres_solve_d.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_25_3eigs.csv"));
    EXPECT_LE((gmres_solve_d.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}
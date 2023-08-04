#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <string>
#include <iostream>

using Eigen::MatrixXd;

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;

class GMRESDoubleTest: public TestBase {};

TEST_F(GMRESDoubleTest, SolveConvDiff64) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv");
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, u_dbl);

    gmres_solve_d.solve(64, conv_tol_dbl);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleTest, SolveConvDiff256) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv");
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, u_dbl);

    gmres_solve_d.solve(256, conv_tol_dbl);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_b.csv");
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, u_dbl);

    gmres_solve_d.solve(1024, conv_tol_dbl);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleTest, SolveRand20) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "A_20_rand.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "b_20_rand.csv");
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, u_dbl);

    gmres_solve_d.solve(20, conv_tol_dbl);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(solve_matrix_dir + "x_20_rand.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(GMRESDoubleTest, Solve3Eigs) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "A_25_3eigs.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "b_25_3eigs.csv");
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, u_dbl);

    gmres_solve_d.solve(3, conv_tol_dbl);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(solve_matrix_dir + "x_25_3eigs.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}
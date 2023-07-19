#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;

class GMRESDoubleTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 4*pow(2, -52); // Set as 4 times machines epsilon
        double convergence_tolerance = 1e-11;

};

TEST_F(GMRESDoubleTest, SolveConvDiff64) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(64, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, x_0, double_tolerance);

    gmres_solve_d.solve(64, convergence_tolerance);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), convergence_tolerance);

    // Check that matches MATLAB gmres solution within the difference of twice convergence_tolerance
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(matrix_dir + "conv_diff_64_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*convergence_tolerance);

}

TEST_F(GMRESDoubleTest, SolveConvDiff256) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(256, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, x_0, double_tolerance);

    gmres_solve_d.solve(256, convergence_tolerance);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), convergence_tolerance);

    // Check that matches MATLAB gmres solution within the difference of twice convergence_tolerance
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(matrix_dir + "conv_diff_256_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*convergence_tolerance);

}

TEST_F(GMRESDoubleTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_1024_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_1024_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(1024, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, x_0, double_tolerance);

    gmres_solve_d.solve(1024, convergence_tolerance);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), convergence_tolerance);

    // Check that matches MATLAB gmres solution within the difference of twice convergence_tolerance
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(matrix_dir + "conv_diff_1024_x.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*convergence_tolerance);

}

TEST_F(GMRESDoubleTest, SolveRand20) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_20_rand.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_20_rand.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(20, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, x_0, double_tolerance);

    gmres_solve_d.solve(20, convergence_tolerance);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), convergence_tolerance);

    // Check that matches MATLAB gmres solution within the difference of twice convergence_tolerance
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(matrix_dir + "x_20_rand.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*convergence_tolerance);

}

TEST_F(GMRESDoubleTest, Solve3Eigs) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_25_3eigs.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_25_3eigs.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(25, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<double> gmres_solve_d(A, b, x_0, double_tolerance);

    gmres_solve_d.solve(3, convergence_tolerance);
    gmres_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_d.check_converged());
    EXPECT_LE(gmres_solve_d.get_relres(), convergence_tolerance);

    // Check that matches MATLAB gmres solution within the difference of twice convergence_tolerance
    Matrix<double, Dynamic, Dynamic> x_test = read_matrix_csv<double>(matrix_dir + "x_25_3eigs.csv");
    EXPECT_LE((gmres_solve_d.x - x_test).norm()/(x_test.norm()), 2*convergence_tolerance);

}
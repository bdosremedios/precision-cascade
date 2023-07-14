#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/Jacobi.h"

using read_matrix::read_matrix_csv;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;

class JacobiTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";

};

TEST_F(JacobiTest, SolveConvDiff64) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(64, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-12;

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(1000, tol);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    double rel_res = (b - A*jacobi_solve_d.soln()).norm()/r_0.norm();
    EXPECT_LE(rel_res, tol);

}

TEST_F(JacobiTest, SolveConvDiff256_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(256, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-2;

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(300, tol);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    double rel_res = (b - A*jacobi_solve_d.soln()).norm()/r_0.norm();
    EXPECT_LE(rel_res, tol);
    
}

#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/Jacobi.h"

using read_matrix::read_matrix_csv;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;

using std::string;
using std::cout, std::endl;

class JacobiTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        int max_iter = 2000;

};

TEST_F(JacobiTest, SolveConvDiff64_Double) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(64, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-10;

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(max_iter, tol);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff256_Double_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(256, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-10;

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(max_iter, tol);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), tol);
    
}

TEST_F(JacobiTest, SolveConvDiff64_Single) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-5;

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(max_iter, tol);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff256_Single_LONGRUNTIME) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(256, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-5;

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(max_iter, tol);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff64_SingleFailBeyondEpsilon) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-8;

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(400, tol);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_FALSE(jacobi_solve_s.check_converged());
    EXPECT_GT(jacobi_solve_s.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff64_Half) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(64, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 0.0997/2;

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(max_iter, tol);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff256_Half_LONGRUNTIME) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(256, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 0.0997/2;

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(max_iter, tol);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), tol);

}

TEST_F(JacobiTest, SolveConvDiff64_HalfFailBeyondEpsilon) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(64, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-4;

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(300, tol);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_FALSE(jacobi_solve_h.check_converged());
    EXPECT_GT(jacobi_solve_h.get_relres(), tol);

}

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/Jacobi.h"

#include <string>

using read_matrix::read_matrix_csv;

using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using Eigen::MatrixXf;
using Eigen::MatrixXd;

using std::string;
using std::cout, std::endl;

class JacobiTest: public TestBase {

    public:
        int max_iter = 2000;
        int fail_iter = 400;

};

TEST_F(JacobiTest, SolveConvDiff64_Double) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv");

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(max_iter, conv_tol_dbl);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), conv_tol_dbl);

}

TEST_F(JacobiTest, SolveConvDiff256_Double_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv");

    JacobiSolve<double> jacobi_solve_d(A, b);
    jacobi_solve_d.solve(max_iter, conv_tol_dbl);
    jacobi_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), conv_tol_dbl);
    
}

TEST_F(JacobiTest, SolveConvDiff64_Single) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_b.csv");

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(max_iter, conv_tol_sgl);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), conv_tol_sgl);

}

TEST_F(JacobiTest, SolveConvDiff256_Single_LONGRUNTIME) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_256_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_256_b.csv");

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(max_iter, conv_tol_sgl);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), conv_tol_sgl);

}

TEST_F(JacobiTest, SolveConvDiff64_SingleFailBeyondEpsilon) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_b.csv");

    JacobiSolve<float> jacobi_solve_s(A, b);
    jacobi_solve_s.solve(fail_iter, 0.1*u_sgl);
    jacobi_solve_s.view_relres_plot("log");
    
    EXPECT_FALSE(jacobi_solve_s.check_converged());
    EXPECT_GT(jacobi_solve_s.get_relres(), 0.1*u_sgl);

}

TEST_F(JacobiTest, SolveConvDiff64_Half) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv");

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(max_iter, conv_tol_hlf);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), conv_tol_hlf);

}

TEST_F(JacobiTest, SolveConvDiff256_Half_LONGRUNTIME) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_b.csv");

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(max_iter, conv_tol_hlf);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), conv_tol_hlf);

}

TEST_F(JacobiTest, SolveConvDiff64_HalfFailBeyondEpsilon) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv");

    JacobiSolve<half> jacobi_solve_h(A, b);
    jacobi_solve_h.solve(fail_iter, 0.1*u_sgl);
    jacobi_solve_h.view_relres_plot("log");
    
    EXPECT_FALSE(jacobi_solve_h.check_converged());
    EXPECT_GT(jacobi_solve_h.get_relres(), 0.1*u_sgl);

}

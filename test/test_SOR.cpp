#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/SOR.h"

using read_matrix::read_matrix_csv;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;

using std::string;
using std::cout, std::endl;

class GaussSeidelTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        int max_iter = 1000;

};

TEST_F(GaussSeidelTest, SolveConvDiff64_Double) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(64, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-10;

    SORSolve<double> SOR_solve_d(A, b, 1);
    SOR_solve_d.solve(max_iter, tol);
    SOR_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_d.check_converged());
    EXPECT_LE(SOR_solve_d.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff256_Double_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<double, Dynamic, 1> x_0 = MatrixXd::Ones(256, 1);
    Matrix<double, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-10;

    SORSolve<double> SOR_solve_d(A, b, 1);
    SOR_solve_d.solve(max_iter, tol);
    SOR_solve_d.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_d.check_converged());
    EXPECT_LE(SOR_solve_d.get_relres(), tol);
    
}

TEST_F(GaussSeidelTest, SolveConvDiff64_Single) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-5;

    SORSolve<float> SOR_solve_s(A, b, 1);
    SOR_solve_s.solve(max_iter, tol);
    SOR_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_s.check_converged());
    EXPECT_LE(SOR_solve_s.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff256_Single_LONGRUNTIME) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(256, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-5;

    SORSolve<float> SOR_solve_s(A, b, 1);
    SOR_solve_s.solve(max_iter, tol);
    SOR_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_s.check_converged());
    EXPECT_LE(SOR_solve_s.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff64_SingleFailBeyondEpsilon) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-8;

    SORSolve<float> SOR_solve_s(A, b, 1);
    SOR_solve_s.solve(max_iter, tol);
    SOR_solve_s.view_relres_plot("log");
    
    EXPECT_FALSE(SOR_solve_s.check_converged());
    EXPECT_GT(SOR_solve_s.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff64_Half) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(64, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 0.0997/2;

    SORSolve<half> SOR_solve_h(A, b, static_cast<half>(1));
    SOR_solve_h.solve(max_iter, tol);
    SOR_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_h.check_converged());
    EXPECT_LE(SOR_solve_h.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff256_Half_LONGRUNTIME) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(256, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 0.0997/2;

    SORSolve<half> SOR_solve_h(A, b, static_cast<half>(1));
    SOR_solve_h.solve(max_iter, tol);
    SOR_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(SOR_solve_h.check_converged());
    EXPECT_LE(SOR_solve_h.get_relres(), tol);

}

TEST_F(GaussSeidelTest, SolveConvDiff64_HalfFailBeyondEpsilon) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(64, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    double tol = 1e-4;

    SORSolve<half> SOR_solve_h(A, b, static_cast<half>(1));
    SOR_solve_h.solve(max_iter, tol);
    SOR_solve_h.view_relres_plot("log");
    
    EXPECT_FALSE(SOR_solve_h.check_converged());
    EXPECT_GT(SOR_solve_h.get_relres(), tol);

}

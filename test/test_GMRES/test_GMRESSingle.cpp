#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;
using Eigen::MatrixXf;
using std::string;
using std::cout, std::endl;

class GMRESSingleTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        float single_tolerance = 4*pow(2, -23); // Set as 4 times machines epsilon
        double convergence_tolerance = 1e-5;

};

TEST_F(GMRESSingleTest, SolveConvDiff64) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(64, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

}

TEST_F(GMRESSingleTest, SolveConvDiff256) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(256, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(256, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

}

TEST_F(GMRESSingleTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_1024_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_1024_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(1024, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(1024, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

}

TEST_F(GMRESSingleTest, SolveRand20) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "A_20_rand.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "b_20_rand.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(20, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(20, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

}

TEST_F(GMRESSingleTest, Solve3Eigs) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "A_25_3eigs.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "b_25_3eigs.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(25, 1);
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(3, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

}

TEST_F(GMRESSingleTest, DivergeBeyondSingleCapabilities) {
    
    Matrix<float, Dynamic, Dynamic> A = read_matrix_csv<float>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<float, Dynamic, Dynamic> b = read_matrix_csv<float>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<float, Dynamic, 1> x_0 = MatrixXf::Ones(64, 1); 
    Matrix<float, Dynamic, 1> r_0 = b - A*x_0;

    // Check convergence under single capabilities
    GMRESSolveTestingMock<float, float> gmres_solve_s(A, b, x_0, single_tolerance);

    gmres_solve_s.solve(128, convergence_tolerance);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*convergence_tolerance);

    // Check divergence beyond single capability of the single machine epsilon
    GMRESSolveTestingMock<float, float> gmres_solve_s_to_fail(A, b, x_0, single_tolerance);
    gmres_solve_s_to_fail.solve(128, 1e-8);
    gmres_solve_s_to_fail.view_relres_plot("log");
    
    EXPECT_FALSE(gmres_solve_s_to_fail.check_converged());
    EXPECT_GT(gmres_solve_s_to_fail.get_relres(), 2e-8);

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include "../test.h"

#include <string>
#include <iostream>

using Eigen::MatrixXf;

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;

class GMRESSingleTest: public TestBase {};

TEST_F(GMRESSingleTest, SolveConvDiff64) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(64, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleTest, SolveConvDiff256) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(256, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(1024, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleTest, SolveRand20) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(20, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleTest, Solve3Eigs) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(25, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_EQ(gmres_solve_s.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

}

TEST_F(GMRESSingleTest, DivergeBeyondSingleCapabilities) {
    
    Matrix<float, Dynamic, Dynamic> A(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<float, Dynamic, 1> b(read_matrix_csv<float>(solve_matrix_dir + "conv_diff_64_b.csv"));

    // Check convergence under single capabilities
    GMRESSolveTestingMock<float> gmres_solve_s(A, b, u_sgl);

    gmres_solve_s.solve(128, conv_tol_sgl);
    gmres_solve_s.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_s.check_converged());
    EXPECT_LE(gmres_solve_s.get_relres(), 2*conv_tol_sgl);

    // Check divergence beyond single capability of the single machine epsilon
    GMRESSolveTestingMock<float> gmres_solve_s_to_fail(A, b, u_sgl);
    gmres_solve_s_to_fail.solve(128, 0.1*u_sgl);
    gmres_solve_s_to_fail.view_relres_plot("log");
    
    EXPECT_FALSE(gmres_solve_s_to_fail.check_converged());
    EXPECT_GT(gmres_solve_s_to_fail.get_relres(), 0.1*u_sgl);

}
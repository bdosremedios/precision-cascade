#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;
using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using std::string;
using std::cout, std::endl;

class GMRESHalfTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        half half_tolerance = static_cast<half>(4*pow(2, -10)); // Set as 4 times machines epsilon
        double convergence_tolerance = 1e-2;

};

TEST_F(GMRESHalfTest, SolveConvDiff64) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_64_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(64, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<half> gmres_solve_h(A, b, x_0, half_tolerance);

    gmres_solve_h.solve(64, convergence_tolerance);
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    double rel_res = (b - A*gmres_solve_h.soln()).norm()/r_0.norm();
    EXPECT_LE(rel_res, 2*convergence_tolerance);

}

TEST_F(GMRESHalfTest, SolveConvDiff256) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_256_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_256_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(256, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<half> gmres_solve_h(A, b, x_0, half_tolerance);

    gmres_solve_h.solve(256, 4*convergence_tolerance);
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    half rel_res = (b - A*gmres_solve_h.soln()).norm()/r_0.norm();
    EXPECT_LE(rel_res, 4*2*convergence_tolerance);

}

TEST_F(GMRESHalfTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<half, Dynamic, Dynamic> A = read_matrix_csv<half>(matrix_dir + "conv_diff_1024_A.csv");
    Matrix<half, Dynamic, Dynamic> b = read_matrix_csv<half>(matrix_dir + "conv_diff_1024_b.csv");
    Matrix<half, Dynamic, 1> x_0 = MatrixXh::Ones(1024, 1);
    Matrix<half, Dynamic, 1> r_0 = b - A*x_0;
    GMRESSolveTestingMock<half> gmres_solve_h(A, b, x_0, half_tolerance);

    gmres_solve_h.solve(1024, 16*convergence_tolerance);
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    double rel_res = (b - A*gmres_solve_h.soln()).norm()/r_0.norm();
    EXPECT_LE(rel_res, 16*2*convergence_tolerance);

}
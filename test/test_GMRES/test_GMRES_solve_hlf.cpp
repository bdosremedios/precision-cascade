#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <iostream>

using Eigen::Matrix, Eigen::Dynamic, Eigen::half;

using read_matrix::read_matrix_csv;

using std::cout, std::endl;

class GMRESSolveHalfTest: public TestBase {
    
    public:
        double large_matrix_error_mod_stag = 4.; // Modification to account for stagnation for
                                                 // error accumulation in larger matrix sizes

};

TEST_F(GMRESSolveHalfTest, SolveConvDiff64) {

    constexpr int n(64);
    Matrix<half, n, n> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), n, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveConvDiff256) {

    constexpr int n(256);
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolve<half> gmres_solve_h(
        A, b, static_cast<half>(u_hlf), n, large_matrix_error_mod_stag*conv_tol_hlf
    );

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveConvDiff1024_LONGRUNTIME) {

    constexpr int n(1024);
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolve<half> gmres_solve_h(
        A, b, static_cast<half>(u_hlf), n, large_matrix_error_mod_stag*conv_tol_hlf
    );

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveRand20) {

    constexpr int n(20);
    Matrix<half, n, n> A(read_matrix_csv<half>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), n, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, Solve3Eigs) {

    constexpr int n(25);
    Matrix<half, n, n> A(read_matrix_csv<half>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), n, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");

    EXPECT_EQ(gmres_solve_h.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, DivergeBeyondHalfCapabilities) {

    constexpr int n(64);
    Matrix<half, n, n> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<half, n, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv"));

    // Check convergence under half capabilities
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), n, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

    // Check divergence beyond single capability of the single machine epsilon
    GMRESSolve<half> gmres_solve_h_to_fail(A, b, static_cast<half>(u_hlf), n, 0.1*u_hlf);
    gmres_solve_h_to_fail.solve();
    gmres_solve_h_to_fail.view_relres_plot("log");
    
    EXPECT_FALSE(gmres_solve_h_to_fail.check_converged());
    EXPECT_GT(gmres_solve_h_to_fail.get_relres(), 0.1*u_hlf);

}

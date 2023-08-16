#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <string>
#include <iostream>

using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;

using read_matrix::read_matrix_csv;

using std::string;
using std::cout, std::endl;

class GMRESHalfTest: public TestBase {
    
    public:
        int max_iter = 100;
        double large_matrix_error_mod_stag = 4.; // Modification to account for stagnation for
                                                 // error accumulation in larger matrix sizes

};

TEST_F(GMRESHalfTest, SolveConvDiff64) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), 64, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESHalfTest, SolveConvDiff256) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolve<half> gmres_solve_h(
        A, b, static_cast<half>(u_hlf), max_iter, large_matrix_error_mod_stag*conv_tol_hlf
    );

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESHalfTest, SolveConvDiff1024_LONGRUNTIME) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolve<half> gmres_solve_h(
        A, b, static_cast<half>(u_hlf), max_iter, large_matrix_error_mod_stag*conv_tol_hlf
    );

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESHalfTest, SolveRand20) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), 20, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESHalfTest, Solve3Eigs) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), 25, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    // TODO: Figure out better check for 3eig since convergence tolerance
    //       isnt reached
    EXPECT_EQ(gmres_solve_h.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESHalfTest, DivergeBeyondHalfCapabilities) {
    
    Matrix<half, Dynamic, Dynamic> A(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<half, Dynamic, 1> b(read_matrix_csv<half>(solve_matrix_dir + "conv_diff_64_b.csv"));

    // Check convergence under half capabilities
    GMRESSolve<half> gmres_solve_h(A, b, static_cast<half>(u_hlf), 128, conv_tol_hlf);

    gmres_solve_h.solve();
    gmres_solve_h.view_relres_plot("log");
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

    // Check divergence beyond single capability of the single machine epsilon
    GMRESSolve<half> gmres_solve_h_to_fail(A, b, static_cast<half>(u_hlf), 128, 0.1*u_hlf);
    gmres_solve_h_to_fail.solve();
    gmres_solve_h_to_fail.view_relres_plot("log");
    
    EXPECT_FALSE(gmres_solve_h_to_fail.check_converged());
    EXPECT_GT(gmres_solve_h_to_fail.get_relres(), 0.1*u_hlf);

}

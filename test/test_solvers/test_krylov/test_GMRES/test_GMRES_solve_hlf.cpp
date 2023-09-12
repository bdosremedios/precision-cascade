#include "../../../test.h"

#include "solvers/krylov/GMRES.h"

class GMRESSolveHalfTest: public TestBase {
    
    public:

        double large_matrix_error_mod_stag = 4.; // Modification to account for stagnation for
                                                 // error accumulation in larger matrix sizes
        
        SolveArgPkg hlf_args;
        SolveArgPkg large_m_modded_hlf_args;
    
        void SetUp() {

            hlf_args.target_rel_res = conv_tol_hlf;
            large_m_modded_hlf_args.target_rel_res = large_matrix_error_mod_stag*conv_tol_hlf;
        
        }

};

TEST_F(GMRESSolveHalfTest, SolveConvDiff64) {

    constexpr int n(64);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveConvDiff256) {

    constexpr int n(256);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, large_m_modded_hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveConvDiff1024_LONGRUNTIME) {

    constexpr int n(1024);
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_1024_b.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, large_m_modded_hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*large_matrix_error_mod_stag*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, SolveRand20) {

    constexpr int n(20);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_20_rand.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_20_rand.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, Solve3Eigs) {

    constexpr int n(25);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_3eigs.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_3eigs.csv"));
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }

    EXPECT_EQ(gmres_solve_h.get_iteration(), 3);
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

}

TEST_F(GMRESSolveHalfTest, DivergeBeyondHalfCapabilities) {

    constexpr int n(64);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    // Check convergence under double capabilities
    GMRESSolve<half> gmres_solve_h(A, b, u_hlf, hlf_args);

    gmres_solve_h.solve();
    if (*show_plots) { gmres_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(gmres_solve_h.check_converged());
    EXPECT_LE(gmres_solve_h.get_relres(), 2*conv_tol_hlf);

    // Check divergence beyond single capability of the single machine epsilon
    SolveArgPkg fail_args; fail_args.target_rel_res = 0.1*u_hlf;
    GMRESSolve<half> gmres_solve_h_to_fail(A, b, u_hlf, fail_args);
    gmres_solve_h_to_fail.solve();
    if (*show_plots) { gmres_solve_h_to_fail.view_relres_plot("log"); }
    
    EXPECT_FALSE(gmres_solve_h_to_fail.check_converged());
    EXPECT_GT(gmres_solve_h_to_fail.get_relres(), 0.1*u_hlf);

}

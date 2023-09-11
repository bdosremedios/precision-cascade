#include "../../test.h"

#include "solvers/stationary/Jacobi.h"

class JacobiTest: public TestBase {

    public:

        SolveArgPkg success_args;
        SolveArgPkg fail_args;

        void SetUp() {

            success_args = SolveArgPkg();
            success_args.max_iter = 2000;

            fail_args = SolveArgPkg();
            fail_args.max_iter = 400;

        }

};

TEST_F(JacobiTest, SolveConvDiff64_Double) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    success_args.target_rel_res = conv_tol_dbl;
    JacobiSolve<double> jacobi_solve_d(A, b, success_args);
    jacobi_solve_d.solve();
    if (show_plots) { jacobi_solve_d.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), conv_tol_dbl);

}

TEST_F(JacobiTest, SolveConvDiff256_Double_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    success_args.target_rel_res = conv_tol_dbl;
    JacobiSolve<double> jacobi_solve_d(A, b, success_args);
    jacobi_solve_d.solve();
    if (show_plots) { jacobi_solve_d.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_d.check_converged());
    EXPECT_LE(jacobi_solve_d.get_relres(), conv_tol_dbl);
    
}

TEST_F(JacobiTest, SolveConvDiff64_Single) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    success_args.target_rel_res = conv_tol_sgl;
    JacobiSolve<float> jacobi_solve_s(A, b, success_args);
    jacobi_solve_s.solve();
    if (show_plots) { jacobi_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), conv_tol_sgl);

}

TEST_F(JacobiTest, SolveConvDiff256_Single_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    success_args.target_rel_res = conv_tol_sgl;
    JacobiSolve<float> jacobi_solve_s(A, b, success_args);
    jacobi_solve_s.solve();
    if (show_plots) { jacobi_solve_s.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_s.check_converged());
    EXPECT_LE(jacobi_solve_s.get_relres(), conv_tol_sgl);

}

TEST_F(JacobiTest, SolveConvDiff64_SingleFailBeyondEpsilon) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    fail_args.target_rel_res = 0.1*u_sgl;
    JacobiSolve<float> jacobi_solve_s(A, b, fail_args);
    jacobi_solve_s.solve();
    if (show_plots) { jacobi_solve_s.view_relres_plot("log"); }
    
    EXPECT_FALSE(jacobi_solve_s.check_converged());
    EXPECT_GT(jacobi_solve_s.get_relres(), 0.1*u_sgl);

}

TEST_F(JacobiTest, SolveConvDiff64_Half) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    success_args.target_rel_res = conv_tol_hlf;
    JacobiSolve<half> jacobi_solve_h(A, b, success_args);
    jacobi_solve_h.solve();
    if (show_plots) { jacobi_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), conv_tol_hlf);

}

TEST_F(JacobiTest, SolveConvDiff256_Half_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    success_args.target_rel_res = conv_tol_hlf;
    JacobiSolve<half> jacobi_solve_h(A, b, success_args);
    jacobi_solve_h.solve();
    if (show_plots) { jacobi_solve_h.view_relres_plot("log"); }
    
    EXPECT_TRUE(jacobi_solve_h.check_converged());
    EXPECT_LE(jacobi_solve_h.get_relres(), conv_tol_hlf);

}

TEST_F(JacobiTest, SolveConvDiff64_HalfFailBeyondEpsilon) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    fail_args.target_rel_res = 0.1*u_sgl;
    JacobiSolve<half> jacobi_solve_h(A, b, fail_args);
    jacobi_solve_h.solve();
    if (show_plots) { jacobi_solve_h.view_relres_plot("log"); }
    
    EXPECT_FALSE(jacobi_solve_h.check_converged());
    EXPECT_GT(jacobi_solve_h.get_relres(), 0.1*u_sgl);

}

#include "../../test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase {

    public:

        SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_dbl);
        SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_sgl);
        SolveArgPkg sgl_fail_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_sgl);
        SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_hlf);
        SolveArgPkg hlf_fail_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_hlf);

};

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff64) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<double> fp_gmres_ir_solve(A, b, u_dbl, dbl_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_dbl);

}

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    FP_GMRES_IR_Solve<double> fp_gmres_ir_solve(A, b, u_dbl, dbl_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_dbl);

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff64) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<float> fp_gmres_ir_solve(A, b, u_dbl, sgl_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_sgl);

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff256) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    FP_GMRES_IR_Solve<float> fp_gmres_ir_solve(A, b, u_dbl, sgl_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_sgl);

}

TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff64) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<half> fp_gmres_ir_solve(A, b, u_dbl, hlf_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_hlf);

}

TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff256) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    FP_GMRES_IR_Solve<half> fp_gmres_ir_solve(A, b, u_dbl, hlf_GMRES_IR_args);

    fp_gmres_ir_solve.solve();

    if (*show_plots) { fp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(fp_gmres_ir_solve.check_converged());
    EXPECT_LE(fp_gmres_ir_solve.get_relres(), conv_tol_hlf);

}
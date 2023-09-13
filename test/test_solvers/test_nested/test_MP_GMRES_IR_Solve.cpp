#include "../../test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class MP_GMRES_IR_Test: public TestBase {

    public:

        SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_dbl);

};

TEST_F(MP_GMRES_IR_Test, ConvergenceTest_ConvDiff64) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    SimpleConstantThreshold mp_gmres_ir_solve(A, b, u_dbl, dbl_GMRES_IR_args);

    mp_gmres_ir_solve.solve();

    if (*show_plots) { mp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(mp_gmres_ir_solve.check_converged());
    EXPECT_LE(mp_gmres_ir_solve.get_relres(), conv_tol_dbl);

}

TEST_F(MP_GMRES_IR_Test, ConvergenceTest_ConvDiff256) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    SimpleConstantThreshold mp_gmres_ir_solve(A, b, u_dbl, dbl_GMRES_IR_args);

    mp_gmres_ir_solve.solve();

    if (*show_plots) { mp_gmres_ir_solve.view_relres_plot("log"); }

    EXPECT_TRUE(mp_gmres_ir_solve.check_converged());
    EXPECT_LE(mp_gmres_ir_solve.get_relres(), conv_tol_dbl);

}
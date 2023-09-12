#include "../../test.h"

#include "solvers/nested/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase {

    public:

        SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(10, 10, conv_tol_dbl);

};

TEST_F(FP_GMRES_IR_Test, BasicTest) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<double> fp_gmres_ir_solve_d(A, b, u_dbl, dbl_GMRES_IR_args);

    fp_gmres_ir_solve_d.solve();

    if (*show_plots) { fp_gmres_ir_solve_d.view_relres_plot("log"); }

}

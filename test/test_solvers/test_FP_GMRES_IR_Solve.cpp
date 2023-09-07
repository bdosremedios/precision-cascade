#include "../test.h"

#include "solvers/nested/FP_GMRES_IR.h"

class FP_GMRES_IRTest: public TestBase {

    // public:
    //     int max_iter = 2000;
    //     int fail_iter = 400;

};

TEST_F(FP_GMRES_IRTest, BasicTest) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<double> fp_gmres_ir_solve_d(A, b, u_dbl, 10, 20, conv_tol_dbl);
    fp_gmres_ir_solve_d.solve();
    fp_gmres_ir_solve_d.view_relres_plot("log");

}

#include "../../test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class MP_GMRES_IR_SolveTest: public TestBase
{
public:

    SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_dbl);

    template <template <typename> typename M>
    void SolveTest(
        const string &A_file_path,
        const string &b_file_path
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);

        SimpleConstantThreshold<M> mp_gmres_ir_solve(A, b, u_dbl, dbl_GMRES_IR_args);

        mp_gmres_ir_solve.solve();

        if (*show_plots) { mp_gmres_ir_solve.view_relres_plot("log"); }

        EXPECT_TRUE(mp_gmres_ir_solve.check_converged());
        EXPECT_LE(mp_gmres_ir_solve.get_relres(), conv_tol_dbl);

    }

};

TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff64_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv"
    );
}
TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff64_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv"
    );
}

TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff256_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv"
    );
}
TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff256_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv"
    );
}
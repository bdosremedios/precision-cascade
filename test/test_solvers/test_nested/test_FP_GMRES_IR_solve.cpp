#include "../../test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase
{
public:

    SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_dbl);
    SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_sgl);
    SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_hlf);
    
    template <template <typename> typename M, typename T>
    void SolveTest(
        const string &A_file_path,
        const string &b_file_path,
        const SolveArgPkg &args,
        const double &u,
        const double &conv_tol
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, T> lin_sys(A, b);

        FP_GMRES_IR_Solve<M, T> gmres_ir(lin_sys, u, args);

        gmres_ir.solve();

        if (*show_plots) { gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(gmres_ir.check_converged());
        EXPECT_LE(gmres_ir.get_relres(), conv_tol);

    }

};

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff64_Dense) {
    SolveTest<MatrixDense, double>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv",
        dbl_GMRES_IR_args, u_dbl, conv_tol_dbl
    );
}
// TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff64_Sparse) {
//     SolveTest<MatrixSparse, double>(
//         solve_matrix_dir + "conv_diff_64_A.csv",
//         solve_matrix_dir + "conv_diff_64_b.csv",
//         dbl_GMRES_IR_args, u_dbl, conv_tol_dbl
//     );
// }

// TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256_Dense) {
//     SolveTest<MatrixDense, double>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         dbl_GMRES_IR_args, u_dbl, conv_tol_dbl
//     );
// }
// TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256_Sparse) {
//     SolveTest<MatrixSparse, double>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         dbl_GMRES_IR_args, u_dbl, conv_tol_dbl
//     );
// }

// TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff64_Dense) {
//     SolveTest<MatrixDense, float>(
//         solve_matrix_dir + "conv_diff_64_A.csv",
//         solve_matrix_dir + "conv_diff_64_b.csv",
//         sgl_GMRES_IR_args, u_sgl, conv_tol_sgl
//     );
// }
// TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff64_Sparse) {
//     SolveTest<MatrixSparse, float>(
//         solve_matrix_dir + "conv_diff_64_A.csv",
//         solve_matrix_dir + "conv_diff_64_b.csv",
//         sgl_GMRES_IR_args, u_sgl, conv_tol_sgl
//     );
// }

// TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff256_Dense) {
//     SolveTest<MatrixDense, float>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         sgl_GMRES_IR_args, u_sgl, conv_tol_sgl
//     );
// }
// TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff256_Sparse) {
//     SolveTest<MatrixSparse, float>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         sgl_GMRES_IR_args, u_sgl, conv_tol_sgl
//     );
// }


// TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff64_Dense) {
//     SolveTest<MatrixDense, half>(
//         solve_matrix_dir + "conv_diff_64_A.csv",
//         solve_matrix_dir + "conv_diff_64_b.csv",
//         hlf_GMRES_IR_args, u_hlf, conv_tol_hlf
//     );
// }
// TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff64_Sparse) {
//     SolveTest<MatrixSparse, half>(
//         solve_matrix_dir + "conv_diff_64_A.csv",
//         solve_matrix_dir + "conv_diff_64_b.csv",
//         hlf_GMRES_IR_args, u_hlf, conv_tol_hlf
//     );
// }

// TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff256_Dense) {
//     SolveTest<MatrixDense, half>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         hlf_GMRES_IR_args, u_hlf, conv_tol_hlf
//     );
// }
// TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff256_Sparse) {
//     SolveTest<MatrixSparse, half>(
//         solve_matrix_dir + "conv_diff_256_A.csv",
//         solve_matrix_dir + "conv_diff_256_b.csv",
//         hlf_GMRES_IR_args, u_hlf, conv_tol_hlf
//     );
// }
#include "../../test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_dbl);
    const SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_sgl);
    const SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(40, 10, conv_tol_hlf);
    
    template <template <typename> typename M, typename T>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args,
        const double &conv_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(A_file_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_file_path));
        TypedLinearSystem<M, T> lin_sys(A, b);

        FP_GMRES_IR_Solve<M, T> gmres_ir(lin_sys, Tol<T>::roundoff(), args);

        gmres_ir.solve();

        if (*show_plots) { gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(gmres_ir.check_converged());
        EXPECT_LE(gmres_ir.get_relres(), conv_tol);

    }

};

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff64) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));
    
    SolveTest<MatrixDense, double>(A_path, b_path, dbl_GMRES_IR_args, conv_tol_dbl);
    SolveTest<MatrixSparse, double>(A_path, b_path, dbl_GMRES_IR_args, conv_tol_dbl);

}

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    SolveTest<MatrixDense, double>(A_path, b_path, dbl_GMRES_IR_args, conv_tol_dbl);
    SolveTest<MatrixSparse, double>(A_path, b_path, dbl_GMRES_IR_args, conv_tol_dbl);

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff64) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, float>(A_path, b_path, sgl_GMRES_IR_args, conv_tol_sgl);
    SolveTest<MatrixSparse, float>(A_path, b_path, sgl_GMRES_IR_args, conv_tol_sgl);

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff256) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, float>(A_path, b_path, sgl_GMRES_IR_args, conv_tol_sgl);
    SolveTest<MatrixSparse, float>(A_path, b_path, sgl_GMRES_IR_args, conv_tol_sgl);

}


TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff64) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, half>(A_path, b_path, hlf_GMRES_IR_args, conv_tol_hlf);
    SolveTest<MatrixSparse, half>(A_path, b_path, hlf_GMRES_IR_args, conv_tol_hlf);

}

TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff256) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, half>(A_path, b_path, hlf_GMRES_IR_args, conv_tol_hlf);
    SolveTest<MatrixSparse, half>(A_path, b_path, hlf_GMRES_IR_args, conv_tol_hlf);

}
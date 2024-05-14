#include "../../test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(80, 10, Tol<double>::krylov_conv_tol());
    const SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(80, 10, Tol<float>::krylov_conv_tol());
    const SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(80, 10, Tol<half>::krylov_conv_tol());
    
    template <template <typename> typename M, typename T>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args,
        const double &conv_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path));
        TypedLinearSystem<M, T> lin_sys(A, b);

        FP_GMRES_IR_Solve<M, T> gmres_ir(lin_sys, Tol<T>::roundoff(), args);

        gmres_ir.solve();

        if (*show_plots) { gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(gmres_ir.check_converged());
        EXPECT_LE(gmres_ir.get_relres(), Tol<T>::krylov_conv_tol());

    }

};

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));
    
    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, DoubleConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SingleConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );

}


TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, HalfConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}
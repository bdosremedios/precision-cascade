#include "test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class FP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<double>::krylov_conv_tol()
    );
    const SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<float>::krylov_conv_tol()
    );
    const SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<half>::krylov_conv_tol()
    );
    
    template <template <typename> typename TMatrix, typename TPrecision>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args,
        const double &conv_tol
    ) {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        FP_GMRES_IR_Solve<TMatrix, TPrecision> gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(), args
        );

        gmres_ir.solve();

        if (*show_plots) { gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(gmres_ir.check_converged());
        EXPECT_LE(gmres_ir.get_relres(), Tol<TPrecision>::krylov_conv_tol());

    }

};

TEST_F(FP_GMRES_IR_Test, SolveConvDiff64_Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));
    
    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff256_Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff1024_Double_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args, Tol<double>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff64_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff256_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff1024_Single_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args, Tol<float>::nested_krylov_conv_tol()
    );

}


TEST_F(FP_GMRES_IR_Test, SolveConvDiff64_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff256_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}

TEST_F(FP_GMRES_IR_Test, SolveConvDiff1024_Half_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );
    SolveTest<NoFillMatrixSparse, half>(
        A_path, b_path, hlf_GMRES_IR_args, Tol<half>::nested_krylov_conv_tol()
    );

}
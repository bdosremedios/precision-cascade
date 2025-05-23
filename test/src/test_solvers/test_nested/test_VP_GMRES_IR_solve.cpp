#include "test.h"

#include "solvers/nested/GMRES_IR/VP_GMRES_IR.h"

class VP_GMRES_IR_Solve_Test: public TestBase
{
public:

    SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<double>::nested_krylov_conv_tol()
    );

    template <
        template <template <typename> typename> typename VP_GMRES_Impl,
        template <typename> typename TMatrix
    >
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path
    ) {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        ));

        GenericLinearSystem<TMatrix> lin_sys(A, b);
        VP_GMRES_Impl<TMatrix> vp_gmres_ir_solve(&lin_sys, dbl_GMRES_IR_args);

        vp_gmres_ir_solve.solve();

        if (*show_plots) { vp_gmres_ir_solve.view_relres_plot("log"); }

        EXPECT_TRUE(vp_gmres_ir_solve.check_converged());
        EXPECT_LE(
            vp_gmres_ir_solve.get_relres(),
            Tol<double>::nested_krylov_conv_tol()
        );

    }

};

TEST_F(VP_GMRES_IR_Solve_Test, OuterRestartCount_SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(VP_GMRES_IR_Solve_Test, OuterRestartCount_SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    OuterRestartCount_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(VP_GMRES_IR_Solve_Test, SD_OuterRestartCount_SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<SD_OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<SD_OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(VP_GMRES_IR_Solve_Test, SD_OuterRestartCount_SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<SD_OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<SD_OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_OuterRestartCount_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<SD_OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<SD_OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    RelativeResidualThreshold_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    RelativeResidualThreshold_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    RelativeResidualThreshold_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_RelativeResidualThreshold_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<SD_RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SD_RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_RelativeResidualThreshold_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<SD_RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SD_RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_RelativeResidualThreshold_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<SD_RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SD_RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    CheckStagnation_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    CheckStagnation_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    CheckStagnation_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_CheckStagnation_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<SD_CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<SD_CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_CheckStagnation_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<SD_CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<SD_CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    SD_CheckStagnation_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<SD_CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<SD_CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    StagnationToThreshold_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<StagnationToThreshold, MatrixDense>(A_path, b_path);
    SolveTest<StagnationToThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    StagnationToThreshold_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<StagnationToThreshold, MatrixDense>(A_path, b_path);
    SolveTest<StagnationToThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    VP_GMRES_IR_Solve_Test,
    StagnationToThreshold_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<StagnationToThreshold, MatrixDense>(A_path, b_path);
    SolveTest<StagnationToThreshold, NoFillMatrixSparse>(A_path, b_path);

}
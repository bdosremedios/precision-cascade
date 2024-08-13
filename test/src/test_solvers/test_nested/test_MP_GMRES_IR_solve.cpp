#include "test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class MP_GMRES_IR_SolveTest: public TestBase
{
public:

    SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<double>::nested_krylov_conv_tol()
    );

    template <
        template <template <typename> typename> typename MP_GMRES_Impl,
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
        MP_GMRES_Impl<TMatrix> mp_gmres_ir_solve(&lin_sys, dbl_GMRES_IR_args);

        mp_gmres_ir_solve.solve();

        if (*show_plots) { mp_gmres_ir_solve.view_relres_plot("log"); }

        EXPECT_TRUE(mp_gmres_ir_solve.check_converged());
        EXPECT_LE(
            mp_gmres_ir_solve.get_relres(),
            Tol<double>::nested_krylov_conv_tol()
        );

    }

};

TEST_F(MP_GMRES_IR_SolveTest, OuterRestartCount_SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, OuterRestartCount_SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    OuterRestartCount_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<OuterRestartCount, MatrixDense>(A_path, b_path);
    SolveTest<OuterRestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    RelativeResidualThreshold_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    RelativeResidualThreshold_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    RelativeResidualThreshold_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<RelativeResidualThreshold, MatrixDense>(A_path, b_path);
    SolveTest<RelativeResidualThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    CheckStagnation_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    CheckStagnation_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    CheckStagnation_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<CheckStagnation, MatrixDense>(A_path, b_path);
    SolveTest<CheckStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    ProjectThresholdAfterStagnation_SolveConvDiff64_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<ProjectThresholdAfterStagnation, MatrixDense>(A_path, b_path);
    SolveTest<ProjectThresholdAfterStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    ProjectThresholdAfterStagnation_SolveConvDiff256_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<ProjectThresholdAfterStagnation, MatrixDense>(A_path, b_path);
    SolveTest<ProjectThresholdAfterStagnation, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    ProjectThresholdAfterStagnation_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<ProjectThresholdAfterStagnation, MatrixDense>(A_path, b_path);
    SolveTest<ProjectThresholdAfterStagnation, NoFillMatrixSparse>(A_path, b_path);

}
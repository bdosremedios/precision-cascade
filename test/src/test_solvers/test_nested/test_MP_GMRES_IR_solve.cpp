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

TEST_F(MP_GMRES_IR_SolveTest, SimpleConstantThreshold_SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<SimpleConstantThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SimpleConstantThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, SimpleConstantThreshold_SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<SimpleConstantThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SimpleConstantThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    SimpleConstantThreshold_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<SimpleConstantThreshold, MatrixDense>(A_path, b_path);
    SolveTest<SimpleConstantThreshold, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, RestartCount_SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<RestartCount, MatrixDense>(A_path, b_path);
    SolveTest<RestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, RestartCount_SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<RestartCount, MatrixDense>(A_path, b_path);
    SolveTest<RestartCount, NoFillMatrixSparse>(A_path, b_path);

}

TEST_F(
    MP_GMRES_IR_SolveTest,
    RestartCount_SolveConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<RestartCount, MatrixDense>(A_path, b_path);
    SolveTest<RestartCount, NoFillMatrixSparse>(A_path, b_path);

}
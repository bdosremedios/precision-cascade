#include "../../test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class MP_GMRES_IR_SolveTest: public TestBase
{
public:

    SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(80, 10, Tol<double>::nested_krylov_conv_tol());

    template <template <typename> typename M>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path
    ) {

        M<double> A(read_matrixCSV<M, double>(*handle_ptr, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(*handle_ptr, b_file_path));

        GenericLinearSystem<M> lin_sys(A, b);
        SimpleConstantThreshold<M> mp_gmres_ir_solve(lin_sys, dbl_GMRES_IR_args);

        mp_gmres_ir_solve.solve();

        if (*show_plots) { mp_gmres_ir_solve.view_relres_plot("log"); }

        EXPECT_TRUE(mp_gmres_ir_solve.check_converged());
        EXPECT_LE(mp_gmres_ir_solve.get_relres(), Tol<double>::nested_krylov_conv_tol());

    }

};

TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff64) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path);
    // SolveTest<MatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff256) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path);
    // SolveTest<MatrixSparse>(A_path, b_path);

}

TEST_F(MP_GMRES_IR_SolveTest, ConvergenceTest_ConvDiff1024_LONGRUNTIME) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path);
    // SolveTest<MatrixSparse>(A_path, b_path);

}
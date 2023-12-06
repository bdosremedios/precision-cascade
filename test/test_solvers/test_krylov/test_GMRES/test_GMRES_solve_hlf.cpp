#include "../../../test.h"

#include "solvers/krylov/GMRES.h"

class GMRES_Solve_Half_Test: public TestBase
{
public:

    double large_matrix_error_mod_stag = 4.; // Modification to account for stagnation for
                                             // error accumulation in larger matrix sizes

    template <template <typename> typename M>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double &large_error_mod,
        const bool &check_3_iter
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, half> lin_sys(A, b);

        SolveArgPkg args;
        args.target_rel_res = large_error_mod*conv_tol_hlf;
        GMRESSolve<M, half> gmres_solve(lin_sys, Tol<half>::roundoff(), args);

        gmres_solve.solve();

        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), large_error_mod*conv_tol_hlf);
        if (check_3_iter) { EXPECT_EQ(gmres_solve.get_iteration(), 3); }

    }

    template <template <typename> typename M>
    void FailTest() {

        constexpr int n(64);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));
        TypedLinearSystem<M, half> lin_sys(A, b);

        // Check convergence under single capabilities
        SolveArgPkg args;
        args.target_rel_res = conv_tol_hlf;
        GMRESSolve<M, half> gmres_solve_succeed(lin_sys, Tol<half>::roundoff(), args);

        gmres_solve_succeed.solve();
        if (*show_plots) { gmres_solve_succeed.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve_succeed.check_converged());
        EXPECT_LE(gmres_solve_succeed.get_relres(), conv_tol_hlf);

        // Check divergence beyond single capability of the single machine epsilon
        SolveArgPkg fail_args;
        fail_args.target_rel_res = 0.1*Tol<half>::roundoff();
        GMRESSolve<M, half> gmres_solve_fail(lin_sys, Tol<half>::roundoff(), fail_args);

        gmres_solve_fail.solve();
        if (*show_plots) { gmres_solve_fail.view_relres_plot("log"); }
        
        EXPECT_FALSE(gmres_solve_fail.check_converged());
        EXPECT_GT(gmres_solve_fail.get_relres(), 0.1*Tol<half>::roundoff());

    }

};

TEST_F(GMRES_Solve_Half_Test, SolveConvDiff64_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        1.,
        false
    );
}
TEST_F(GMRES_Solve_Half_Test, SolveConvDiff64_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        1.,
        false
    );
}

TEST_F(GMRES_Solve_Half_Test, SolveConvDiff256_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        large_matrix_error_mod_stag,
        false
    );
}
TEST_F(GMRES_Solve_Half_Test, SolveConvDiff256_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        large_matrix_error_mod_stag,
        false
    );
}

TEST_F(GMRES_Solve_Half_Test, SolveConvDiff1024_Dense_LONGRUNTIME) {
    SolveTest<MatrixDense>(
        solve_matrix_dir / fs::path("conv_diff_1024_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_1024_b.csv"),
        large_matrix_error_mod_stag,
        false
    );
}
TEST_F(GMRES_Solve_Half_Test, SolveConvDiff1024_Sparse_LONGRUNTIME) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir / fs::path("conv_diff_1024_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_1024_b.csv"),
        large_matrix_error_mod_stag,
        false
    );
}

TEST_F(GMRES_Solve_Half_Test, SolveConvDiff20Rand_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir / fs::path("A_20_rand.csv"),
        solve_matrix_dir / fs::path("b_20_rand.csv"),
        1.,
        false
    );
}
TEST_F(GMRES_Solve_Half_Test, SolveConvDiff20Rand_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir / fs::path("A_20_rand.csv"),
        solve_matrix_dir / fs::path("b_20_rand.csv"),
        1.,
        false
    );
}

TEST_F(GMRES_Solve_Half_Test, SolveConvDiff3Eigs_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir / fs::path("A_25_3eigs.csv"),
        solve_matrix_dir / fs::path("b_25_3eigs.csv"),
        1.,
        true
    );
}
TEST_F(GMRES_Solve_Half_Test, SolveConvDiff3Eigs_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir / fs::path("A_25_3eigs.csv"),
        solve_matrix_dir / fs::path("b_25_3eigs.csv"),
        1.,
        true
    );
}

TEST_F(GMRES_Solve_Half_Test, DivergeBeyondSingleCapabilities_Dense) { FailTest<MatrixDense>(); }
TEST_F(GMRES_Solve_Half_Test, DivergeBeyondSingleCapabilities_Sparse) { FailTest<MatrixSparse>(); }
#include "../../../test.h"

#include "solvers/krylov/GMRES.h"

class GMRESSingleSolveTest: public TestBase
{
public:

    template <template <typename> typename M>
    void SolveTest(
        const string &A_file_path,
        const string &b_file_path,
        const bool &check_3_iter
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, float> lin_sys(A, b);

        SolveArgPkg args;
        args.target_rel_res = conv_tol_sgl;
        GMRESSolve<M, float> gmres_solve(lin_sys, u_sgl, args);

        gmres_solve.solve();

        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), conv_tol_sgl);
        if (check_3_iter) { EXPECT_EQ(gmres_solve.get_iteration(), 3); }

    }

    template <template <typename> typename M>
    void FailTest() {

        constexpr int n(64);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "conv_diff_64_A.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "conv_diff_64_b.csv");
        TypedLinearSystem<M, float> lin_sys(A, b);

        // Check convergence under single capabilities
        SolveArgPkg args;
        args.target_rel_res = conv_tol_sgl;
        GMRESSolve<M, float> gmres_solve_succeed(lin_sys, u_sgl, args);

        gmres_solve_succeed.solve();
        if (*show_plots) { gmres_solve_succeed.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve_succeed.check_converged());
        EXPECT_LE(gmres_solve_succeed.get_relres(), conv_tol_sgl);

        // Check divergence beyond single capability of the single machine epsilon
        SolveArgPkg fail_args;
        fail_args.target_rel_res = 0.1*u_sgl;
        GMRESSolve<M, float> gmres_solve_fail(lin_sys, u_sgl, fail_args);

        gmres_solve_fail.solve();
        if (*show_plots) { gmres_solve_fail.view_relres_plot("log"); }
        
        EXPECT_FALSE(gmres_solve_fail.check_converged());
        EXPECT_GT(gmres_solve_fail.get_relres(), 0.1*u_sgl);

    }

};

TEST_F(GMRESSingleSolveTest, SolveConvDiff64_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv",
        false
    );
}

TEST_F(GMRESSingleSolveTest, SolveConvDiff64_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv",
        false
    );
}

TEST_F(GMRESSingleSolveTest, SolveConvDiff256_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv",
        false
    );
}
TEST_F(GMRESSingleSolveTest, SolveConvDiff256_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv",
        false
    );
}

TEST_F(GMRESSingleSolveTest, SolveConvDiff1024_Dense_LONGRUNTIME) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_1024_A.csv",
        solve_matrix_dir + "conv_diff_1024_b.csv",
        false
    );
}
TEST_F(GMRESSingleSolveTest, SolveConvDiff1024_Sparse_LONGRUNTIME) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_1024_A.csv",
        solve_matrix_dir + "conv_diff_1024_b.csv",
        false
    );
}

TEST_F(GMRESSingleSolveTest, SolveConvDiff20Rand_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "A_20_rand.csv",
        solve_matrix_dir + "b_20_rand.csv",
        false
    );
}
TEST_F(GMRESSingleSolveTest, SolveConvDiff20Rand_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "A_20_rand.csv",
        solve_matrix_dir + "b_20_rand.csv",
        false
    );
}

TEST_F(GMRESSingleSolveTest, SolveConvDiff3Eigs_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "A_25_3eigs.csv",
        solve_matrix_dir + "b_25_3eigs.csv",
        true
    );
}
TEST_F(GMRESSingleSolveTest, SolveConvDiff3Eigs_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "A_25_3eigs.csv",
        solve_matrix_dir + "b_25_3eigs.csv",
        true
    );
}

TEST_F(GMRESSingleSolveTest, DivergeBeyondSingleCapabilities_Dense) { FailTest<MatrixDense>(); }
TEST_F(GMRESSingleSolveTest, DivergeBeyondSingleCapabilities_Sparse) { FailTest<MatrixSparse>(); }
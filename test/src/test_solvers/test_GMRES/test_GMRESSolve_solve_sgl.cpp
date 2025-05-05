#include "test.h"

#include "solvers/GMRES/GMRESSolve.h"

class GMRESSolve_Solve_SGL_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const bool &check_3_iter
    ) {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, float> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        args.target_rel_res = Tol<float>::krylov_conv_tol();
        GMRESSolve<TMatrix, float> gmres_solve(
            &typed_lin_sys, args
        );

        gmres_solve.solve();

        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), Tol<float>::krylov_conv_tol());
        if (check_3_iter) { EXPECT_EQ(gmres_solve.get_iteration(), 3); }

    }

    template <template <typename> typename TMatrix>
    void FailTest() {

        constexpr int n(64);
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_A.csv")
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_b.csv")
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, float> typed_lin_sys(&gen_lin_sys);

        // Check convergence under single capabilities
        SolveArgPkg args;
        args.target_rel_res = Tol<float>::krylov_conv_tol();
        GMRESSolve<TMatrix, float> gmres_solve_succeed(
            &typed_lin_sys, args
        );

        gmres_solve_succeed.solve();
        if (*show_plots) { gmres_solve_succeed.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve_succeed.check_converged());
        EXPECT_LE(
            gmres_solve_succeed.get_relres(),
            Tol<float>::krylov_conv_tol()
        );

        // Check divergence beyond single capability of the single machine eps
        SolveArgPkg fail_args;
        fail_args.target_rel_res = 0.1*Tol<float>::roundoff();
        GMRESSolve<TMatrix, float> gmres_solve_fail(
            &typed_lin_sys, fail_args
        );

        gmres_solve_fail.solve();
        if (*show_plots) { gmres_solve_fail.view_relres_plot("log"); }
        
        EXPECT_FALSE(gmres_solve_fail.check_converged());
        EXPECT_GT(gmres_solve_fail.get_relres(), 0.1*Tol<float>::roundoff());

    }

};

TEST_F(GMRESSolve_Solve_SGL_Test, SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, false);

}

TEST_F(GMRESSolve_Solve_SGL_Test, SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, false);

}

TEST_F(GMRESSolve_Solve_SGL_Test, SolveConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    SolveTest<MatrixDense>(A_path, b_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, false);

}

TEST_F(GMRESSolve_Solve_SGL_Test, SolveConvDiff20Rand_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("A_20_rand.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_20_rand.csv"));

    SolveTest<MatrixDense>(A_path, b_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, false);

}

TEST_F(GMRESSolve_Solve_SGL_Test, SolveConvDiff3Eigs_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("A_25_3eigs.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_25_3eigs.csv"));

    SolveTest<MatrixDense>(A_path, b_path, true);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, true);

}

TEST_F(GMRESSolve_Solve_SGL_Test, DivergeBeyondCapabilities_Single_SOLVER) {
    FailTest<MatrixDense>();
    FailTest<NoFillMatrixSparse>();
}
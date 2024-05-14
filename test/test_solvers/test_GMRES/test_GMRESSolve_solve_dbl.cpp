#include "../../test.h"

#include "solvers/GMRES/GMRESSolve.h"

class GMRESSolve_Solve_DBL_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void SolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const fs::path &x_file_path,
        const bool &check_3_iter
    ) {

        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path));
        TypedLinearSystem<M, double> lin_sys(A, b);

        SolveArgPkg args;
        args.target_rel_res = Tol<double>::krylov_conv_tol();
        GMRESSolve<M, double> gmres_solve(lin_sys, Tol<double>::roundoff(), args);

        gmres_solve.solve();

        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), Tol<double>::krylov_conv_tol());
        if (check_3_iter) { EXPECT_EQ(gmres_solve.get_iteration(), 3); }

        // Check that matches MATLAB gmres solution close within conv_tol_dbl
        Vector<double> x_test(read_matrixCSV<Vector, double>(TestBase::bundle, x_file_path));
        EXPECT_LE(
            ((gmres_solve.get_typed_soln() - x_test).norm()/(x_test.norm())).get_scalar(),
            2*Tol<double>::krylov_conv_tol()
        );

    }

};

TEST_F(GMRESSolve_Solve_DBL_Test, SolveConvDiff64_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));
    fs::path x_path(solve_matrix_dir / fs::path("conv_diff_64_x.csv"));

    SolveTest<MatrixDense>(A_path, b_path, x_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, x_path, false);

}

TEST_F(GMRESSolve_Solve_DBL_Test, SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    fs::path x_path(solve_matrix_dir / fs::path("conv_diff_256_x.csv"));

    SolveTest<MatrixDense>(A_path, b_path, x_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, x_path, false);

}

TEST_F(GMRESSolve_Solve_DBL_Test, SolveConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));
    fs::path x_path(solve_matrix_dir / fs::path("conv_diff_1024_x.csv"));

    SolveTest<MatrixDense>(A_path, b_path, x_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, x_path, false);

}

TEST_F(GMRESSolve_Solve_DBL_Test, SolveConvDiff20Rand_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("A_20_rand.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_20_rand.csv"));
    fs::path x_path(solve_matrix_dir / fs::path("x_20_rand.csv"));

    SolveTest<MatrixDense>(A_path, b_path, x_path, false);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, x_path, false);

}

TEST_F(GMRESSolve_Solve_DBL_Test, SolveConvDiff3Eigs_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("A_25_3eigs.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_25_3eigs.csv"));
    fs::path x_path(solve_matrix_dir / fs::path("x_25_3eigs.csv"));

    SolveTest<MatrixDense>(A_path, b_path, x_path, true);
    SolveTest<NoFillMatrixSparse>(A_path, b_path, x_path, true);

}
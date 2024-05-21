#include "../../test.h"

#include <vector>

#include "solvers/stationary/SORSolve.h"

class SOR_Test: public TestBase
{
public:

    std::vector<double> ws{1.25, 1.5, 1.75};

    template <template <typename> typename M, typename T>
    void SolveSuccessTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double conv_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        args.max_iter = 1000;
        args.target_rel_res = conv_tol;
    
        for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

            std::cout << "Testing w=" << *w << std::endl;

            SORSolve<M, T> gauss_seidel_solve(&typed_lin_sys, *w, args);
            gauss_seidel_solve.solve();
            if (*show_plots) { gauss_seidel_solve.view_relres_plot("log"); }
            
            EXPECT_TRUE(gauss_seidel_solve.check_converged());
            EXPECT_LE(gauss_seidel_solve.get_relres(), conv_tol);

        }

    }

    template <template <typename> typename M, typename T>
    void SolveFailTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double fail_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(TestBase::bundle, A_file_path));
        Vector<double> b(read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        args.max_iter = 300;
        args.target_rel_res = fail_tol;

        for (auto w = ws.cbegin(); w != ws.cend(); ++w) {
    
            SORSolve<M, T> gauss_seidel_solve(&typed_lin_sys, *w, args);
            gauss_seidel_solve.solve();
            if (*show_plots) { gauss_seidel_solve.view_relres_plot("log"); }
            
            EXPECT_FALSE(gauss_seidel_solve.check_converged());
            EXPECT_GT(gauss_seidel_solve.get_relres(), fail_tol);

        }

    }

};

TEST_F(SOR_Test, SolveConvDiff64Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, half>(A_path, b_path, Tol<half>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, half>(A_path, b_path, Tol<half>::stationary_conv_tol());

}

TEST_F(SOR_Test, SolveConvDiff256Half_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, half>(A_path, b_path, Tol<half>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, half>(A_path, b_path, Tol<half>::stationary_conv_tol());

}

TEST_F(SOR_Test, SolveConvDiff64Half_FailBeyondCapabilities_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, half>(A_path, b_path, 0.1*Tol<half>::roundoff());
    SolveFailTest<NoFillMatrixSparse, half>(A_path, b_path, 0.1*Tol<half>::roundoff());

}

TEST_F(SOR_Test, SolveConvDiff64Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, float>(A_path, b_path, Tol<float>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, float>(A_path, b_path, Tol<float>::stationary_conv_tol());

}

TEST_F(SOR_Test, SolveConvDiff256Single_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, float>(A_path, b_path, Tol<float>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, float>(A_path, b_path, Tol<float>::stationary_conv_tol());

}

TEST_F(SOR_Test, SolveConvDiff64Single_FailBeyondCapabilities_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, float>(A_path, b_path, 0.1*Tol<float>::roundoff());
    SolveFailTest<NoFillMatrixSparse, float>(A_path, b_path, 0.1*Tol<float>::roundoff());

}

TEST_F(SOR_Test, SolveConvDiff64Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, double>(A_path, b_path, Tol<double>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, double>(A_path, b_path, Tol<double>::stationary_conv_tol());

}

TEST_F(SOR_Test, SolveConvDiff256Double_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, double>(A_path, b_path, Tol<double>::stationary_conv_tol());
    SolveSuccessTest<NoFillMatrixSparse, double>(A_path, b_path, Tol<double>::stationary_conv_tol());

}
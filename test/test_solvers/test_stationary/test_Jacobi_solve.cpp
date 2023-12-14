#include "../../test.h"

#include "solvers/stationary/Jacobi.h"

class Jacobi_Test: public TestBase 
{
public:

    template <template <typename> typename M, typename T>
    void SolveSuccessTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double conv_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(A_file_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_file_path));
        TypedLinearSystem<M, T> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = 2000;
        args.target_rel_res = conv_tol;
    
        JacobiSolve<M, T> jacobi_solve(lin_sys, args);
        jacobi_solve.solve();
        if (*show_plots) { jacobi_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(jacobi_solve.check_converged());
        EXPECT_LE(jacobi_solve.get_relres(), conv_tol);

    }

    template <template <typename> typename M, typename T>
    void SolveFailTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double fail_tol
    ) {

        M<double> A(read_matrixCSV<M, double>(A_file_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_file_path));
        TypedLinearSystem<M, T> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = 400;
        args.target_rel_res = fail_tol;
    
        JacobiSolve<M, T> jacobi_solve(lin_sys, args);
        jacobi_solve.solve();
        if (*show_plots) { jacobi_solve.view_relres_plot("log"); }
        
        EXPECT_FALSE(jacobi_solve.check_converged());
        EXPECT_GT(jacobi_solve.get_relres(), fail_tol);

    }

};

TEST_F(Jacobi_Test, SolveConvDiff64Double) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, double>(A_path, b_path, conv_tol_dbl);
    SolveSuccessTest<MatrixSparse, double>(A_path, b_path, conv_tol_dbl);

}

TEST_F(Jacobi_Test, SolveConvDiff256Double_LONGRUNTIME) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, double>(A_path, b_path, conv_tol_dbl);
    SolveSuccessTest<MatrixSparse, double>(A_path, b_path, conv_tol_dbl);

}

TEST_F(Jacobi_Test, SolveConvDiff64Single) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, float>(A_path, b_path, conv_tol_sgl);
    SolveSuccessTest<MatrixSparse, float>(A_path, b_path, conv_tol_sgl);

}

TEST_F(Jacobi_Test, SolveConvDiff256Single_LONGRUNTIME) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, float>(A_path, b_path, conv_tol_sgl);
    SolveSuccessTest<MatrixSparse, float>(A_path, b_path, conv_tol_sgl);

}

TEST_F(Jacobi_Test, SolveConvDiff64Single_FailBeyondCapabilities) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, float>(A_path, b_path, 0.1*Tol<float>::roundoff());
    SolveFailTest<MatrixSparse, float>(A_path, b_path, 0.1*Tol<float>::roundoff());

}

TEST_F(Jacobi_Test, SolveConvDiff64Half) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, half>(A_path, b_path, conv_tol_hlf);
    SolveSuccessTest<MatrixSparse, half>(A_path, b_path, conv_tol_hlf);

}

TEST_F(Jacobi_Test, SolveConvDiff256Half_LONGRUNTIME) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, half>(A_path, b_path, conv_tol_hlf);
    SolveSuccessTest<MatrixSparse, half>(A_path, b_path, conv_tol_hlf);

}

TEST_F(Jacobi_Test, SolveConvDiff64Half_FailBeyondCapabilities) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, half>(A_path, b_path, 0.1*Tol<half>::roundoff());
    SolveFailTest<MatrixSparse, half>(A_path, b_path, 0.1*Tol<half>::roundoff());

}
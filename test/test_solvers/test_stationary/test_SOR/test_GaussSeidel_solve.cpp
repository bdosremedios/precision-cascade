#include "../../../test.h"

#include "solvers/stationary/SOR.h"

class GaussSeidel_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void SolveSuccessTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double conv_tol
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, T> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = 1000;
        args.target_rel_res = conv_tol;
    
        SORSolve<M, T> gauss_seidel_solve(lin_sys, 1, args);
        gauss_seidel_solve.solve();
        if (*show_plots) { gauss_seidel_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gauss_seidel_solve.check_converged());
        EXPECT_LE(gauss_seidel_solve.get_relres(), conv_tol);

    }

    template <template <typename> typename M, typename T>
    void SolveFailTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double fail_tol
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, T> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = 300;
        args.target_rel_res = fail_tol;
    
        SORSolve<M, T> gauss_seidel_solve(lin_sys, 1, args);
        gauss_seidel_solve.solve();
        if (*show_plots) { gauss_seidel_solve.view_relres_plot("log"); }
        
        EXPECT_FALSE(gauss_seidel_solve.check_converged());
        EXPECT_GT(gauss_seidel_solve.get_relres(), fail_tol);

    }

};

TEST_F(GaussSeidel_Test, SolveConvDiff64Double_Dense) {
    SolveSuccessTest<MatrixDense, double>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_dbl
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff64Double_Sparse) {
    SolveSuccessTest<MatrixSparse, double>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_dbl
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff256Double_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, double>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_dbl
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff256Double_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, double>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_dbl
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff64Single_Dense) {
    SolveSuccessTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_sgl
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff64Single_Sparse) {
    SolveSuccessTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_sgl
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff256Single_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_sgl
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff256Single_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_sgl
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff64Single_FailBeyondCapabilities_Dense) {
    SolveFailTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_sgl
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff64Single_FailBeyondCapabilities_Sparse) {
    SolveFailTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_sgl
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff64Half_Dense) {
    SolveSuccessTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_hlf
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff64Half_Sparse) {
    SolveSuccessTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_hlf
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff256Half_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_hlf
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff256Half_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_hlf
    );
}

TEST_F(GaussSeidel_Test, SolveConvDiff64Half_FailBeyondCapabilities_Dense) {
    SolveFailTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_hlf
    );
}
TEST_F(GaussSeidel_Test, SolveConvDiff64Half_FailBeyondCapabilities_Sparse) {
    SolveFailTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_hlf
    );
}
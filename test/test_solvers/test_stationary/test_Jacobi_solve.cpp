#include "../../test.h"

#include "solvers/stationary/Jacobi.h"

class JacobiTest: public TestBase 
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

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
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

TEST_F(JacobiTest, SolveConvDiff64Double_Dense) {
    SolveSuccessTest<MatrixDense, double>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_dbl
    );
}
TEST_F(JacobiTest, SolveConvDiff64Double_Sparse) {
    SolveSuccessTest<MatrixSparse, double>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_dbl
    );
}

TEST_F(JacobiTest, SolveConvDiff256Double_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, double>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_dbl
    );
}
TEST_F(JacobiTest, SolveConvDiff256Double_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, double>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_dbl
    );
}

TEST_F(JacobiTest, SolveConvDiff64Single_Dense) {
    SolveSuccessTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_sgl
    );
}
TEST_F(JacobiTest, SolveConvDiff64Single_Sparse) {
    SolveSuccessTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_sgl
    );
}

TEST_F(JacobiTest, SolveConvDiff256Single_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_sgl
    );
}
TEST_F(JacobiTest, SolveConvDiff256Single_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_sgl
    );
}

TEST_F(JacobiTest, SolveConvDiff64Single_FailBeyondCapabilities_Dense) {
    SolveFailTest<MatrixDense, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_sgl
    );
}
TEST_F(JacobiTest, SolveConvDiff64Single_FailBeyondCapabilities_Sparse) {
    SolveFailTest<MatrixSparse, float>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_sgl
    );
}

TEST_F(JacobiTest, SolveConvDiff64Half_Dense) {
    SolveSuccessTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_hlf
    );
}
TEST_F(JacobiTest, SolveConvDiff64Half_Sparse) {
    SolveSuccessTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        conv_tol_hlf
    );
}

TEST_F(JacobiTest, SolveConvDiff256Half_Dense_LONGRUNTIME) {
    SolveSuccessTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_hlf
    );
}
TEST_F(JacobiTest, SolveConvDiff256Half_Sparse_LONGRUNTIME) {
    SolveSuccessTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_256_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_256_b.csv"),
        conv_tol_hlf
    );
}

TEST_F(JacobiTest, SolveConvDiff64Half_FailBeyondCapabilities_Dense) {
    SolveFailTest<MatrixDense, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_hlf
    );
}
TEST_F(JacobiTest, SolveConvDiff64Half_FailBeyondCapabilities_Sparse) {
    SolveFailTest<MatrixSparse, half>(
        solve_matrix_dir / fs::path("conv_diff_64_A.csv"),
        solve_matrix_dir / fs::path("conv_diff_64_b.csv"),
        0.1*u_hlf
    );
}

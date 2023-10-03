#include "../../../test.h"

#include "solvers/krylov/GMRES.h"

class GMRESDoubleSolveTest: public TestBase
{
public:

    template <template <typename> typename M>
    void SolveTest(
        const string &A_file_path,
        const string &b_file_path,
        const string &x_file_path,
        const bool &check_3_iter
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        TypedLinearSystem<M, double> lin_sys(A, b);

        SolveArgPkg args;
        args.target_rel_res = conv_tol_dbl;
        GMRESSolve<M, double> gmres_solve(lin_sys, u_dbl, args);

        gmres_solve.solve();

        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), conv_tol_dbl);
        if (check_3_iter) { EXPECT_EQ(gmres_solve.get_iteration(), 3); }

        // Check that matches MATLAB gmres solutionclsoe to conv_tol_dbl
        MatrixVector<double> x_test = read_matrixCSV<MatrixVector, double>(x_file_path);
        EXPECT_LE((gmres_solve.get_typed_soln() - x_test).norm()/(x_test.norm()),
                  4*conv_tol_dbl);

    }

};

TEST_F(GMRESDoubleSolveTest, SolveConvDiff64_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv",
        solve_matrix_dir + "conv_diff_64_x.csv",
        false
    );
}
TEST_F(GMRESDoubleSolveTest, SolveConvDiff64_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_64_A.csv",
        solve_matrix_dir + "conv_diff_64_b.csv",
        solve_matrix_dir + "conv_diff_64_x.csv",
        false
    );
}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff256_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv",
        solve_matrix_dir + "conv_diff_256_x.csv",
        false
    );
}
TEST_F(GMRESDoubleSolveTest, SolveConvDiff256_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_256_A.csv",
        solve_matrix_dir + "conv_diff_256_b.csv",
        solve_matrix_dir + "conv_diff_256_x.csv",
        false
    );
}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff1024_Dense_LONGRUNTIME) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "conv_diff_1024_A.csv",
        solve_matrix_dir + "conv_diff_1024_b.csv",
        solve_matrix_dir + "conv_diff_1024_x.csv",
        false
    );
}
TEST_F(GMRESDoubleSolveTest, SolveConvDiff1024_Sparse_LONGRUNTIME) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "conv_diff_1024_A.csv",
        solve_matrix_dir + "conv_diff_1024_b.csv",
        solve_matrix_dir + "conv_diff_1024_x.csv",
        false
    );
}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff20Rand_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "A_20_rand.csv",
        solve_matrix_dir + "b_20_rand.csv",
        solve_matrix_dir + "x_20_rand.csv",
        false
    );
}
TEST_F(GMRESDoubleSolveTest, SolveConvDiff20Rand_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "A_20_rand.csv",
        solve_matrix_dir + "b_20_rand.csv",
        solve_matrix_dir + "x_20_rand.csv",
        false
    );
}

TEST_F(GMRESDoubleSolveTest, SolveConvDiff3Eigs_Dense) {
    SolveTest<MatrixDense>(
        solve_matrix_dir + "A_25_3eigs.csv",
        solve_matrix_dir + "b_25_3eigs.csv",
        solve_matrix_dir + "x_25_3eigs.csv",
        true
    );
}
TEST_F(GMRESDoubleSolveTest, SolveConvDiff3Eigs_Sparse) {
    SolveTest<MatrixSparse>(
        solve_matrix_dir + "A_25_3eigs.csv",
        solve_matrix_dir + "b_25_3eigs.csv",
        solve_matrix_dir + "x_25_3eigs.csv",
        true
    );
}
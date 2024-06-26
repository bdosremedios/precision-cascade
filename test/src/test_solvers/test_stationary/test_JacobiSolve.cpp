#include "test.h"

#include "solvers/stationary/JacobiSolve.h"

class Jacobi_Test: public TestBase 
{
public:

    template <template <typename> typename TMatrix, typename TPrecision>
    void SolveSuccessTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double conv_tol
    ) {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        args.max_iter = 2000;
        args.target_rel_res = conv_tol;
    
        JacobiSolve<TMatrix, TPrecision> jacobi_solve(&typed_lin_sys, args);
        jacobi_solve.solve();
        if (*show_plots) { jacobi_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(jacobi_solve.check_converged());
        EXPECT_LE(jacobi_solve.get_relres(), conv_tol);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void SolveFailTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const double fail_tol
    ) {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        args.max_iter = 400;
        args.target_rel_res = fail_tol;
    
        JacobiSolve<TMatrix, TPrecision> jacobi_solve(&typed_lin_sys, args);
        jacobi_solve.solve();
        if (*show_plots) { jacobi_solve.view_relres_plot("log"); }
        
        EXPECT_FALSE(jacobi_solve.check_converged());
        EXPECT_GT(jacobi_solve.get_relres(), fail_tol);

    }

};

TEST_F(Jacobi_Test, DivergeBeyondCapabilities_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, float>(
        A_path, b_path, 0.1*Tol<float>::roundoff()
    );
    SolveFailTest<NoFillMatrixSparse, float>(
        A_path, b_path, 0.1*Tol<float>::roundoff()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff64_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, half>(
        A_path, b_path, Tol<half>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, half>(
        A_path, b_path, Tol<half>::stationary_conv_tol()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff256_Half_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, half>(
        A_path, b_path, Tol<half>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, half>(
        A_path, b_path, Tol<half>::stationary_conv_tol()
    );

}

TEST_F(Jacobi_Test, DivergeBeyondCapabilities_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveFailTest<MatrixDense, half>(
        A_path, b_path, 0.1*Tol<half>::roundoff()
    );
    SolveFailTest<NoFillMatrixSparse, half>(
        A_path, b_path, 0.1*Tol<half>::roundoff()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff64_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, float>(
        A_path, b_path, Tol<float>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, float>(
        A_path, b_path, Tol<float>::stationary_conv_tol()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff256_Single_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, float>(
        A_path, b_path, Tol<float>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, float>(
        A_path, b_path, Tol<float>::stationary_conv_tol()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff64_Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_64_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_64_b.csv"));

    SolveSuccessTest<MatrixDense, double>(
        A_path, b_path, Tol<double>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, double>(
        A_path, b_path, Tol<double>::stationary_conv_tol()
    );

}

TEST_F(Jacobi_Test, SolveConvDiff256_Double_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    SolveSuccessTest<MatrixDense, double>(
        A_path, b_path, Tol<double>::stationary_conv_tol()
    );
    SolveSuccessTest<NoFillMatrixSparse, double>(
        A_path, b_path, Tol<double>::stationary_conv_tol()
    );

}
#include "test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Preconditioned_MP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg solve_args = SolveArgPkg(
        80, 10, Tol<double>::krylov_conv_tol()
    );

    template <template <typename> typename TMatrix>
    void PreconditionedMinimumIterTest(
        const SolveArgPkg &args
    ) {

        fs::path A_path(solve_matrix_dir / fs::path("A_inv_45.csv"));
        fs::path Asqr_path(solve_matrix_dir / fs::path("Asqr_inv_45.csv"));
        fs::path Ainv_path(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
        fs::path b_path(solve_matrix_dir / fs::path("b_inv_45.csv"));

        TMatrix<double> A = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_path
        );
        TMatrix<double> Asqr = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, Asqr_path
        );
        Vector<double> b = read_vectorCSV<double>(
            TestBase::bundle, b_path
        );
        TMatrix<double> Ainv = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, Ainv_path
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);

        PrecondArgPkg<TMatrix, double> precond_left_args(
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                Ainv
            )
        );
        SimpleConstantThreshold<TMatrix> precond_left_gmres_ir(
            &gen_lin_sys, args, precond_left_args
        );
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_left_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<TMatrix, double> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, double>>(),
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                Ainv
            )
        );
        SimpleConstantThreshold<TMatrix> precond_right_gmres_ir(
            &gen_lin_sys, args, precond_right_args
        );
        precond_right_gmres_ir.solve();

        if (*show_plots) { precond_right_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_right_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_right_gmres_ir.check_converged());
        EXPECT_LE(precond_right_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<TMatrix, double> precond_symmetric_args(
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                Ainv
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                Ainv
            )
        );
        GenericLinearSystem<TMatrix> gen_lin_sys_Asqr(Asqr, b);

        SimpleConstantThreshold<TMatrix> precond_symmetric_gmres_ir(
            &gen_lin_sys_Asqr, args, precond_symmetric_args
        );
        precond_symmetric_gmres_ir.solve();

        if (*show_plots) { precond_symmetric_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_symmetric_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_symmetric_gmres_ir.check_converged());
        EXPECT_LE(precond_symmetric_gmres_ir.get_relres(), args.target_rel_res);

    }

    template <template <typename> typename TMatrix>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args
    ) {

        TMatrix<double> A = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        );
        Vector<double> b = read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        );

        std::shared_ptr<ILUPreconditioner<TMatrix, double>> ilu_ptr(
            new ILUPreconditioner<TMatrix, double>(A)
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);

        RestartCount<TMatrix> gmres_ir(&gen_lin_sys, args);
        gmres_ir.solve();

        PrecondArgPkg<TMatrix, double> precond_left_args(ilu_ptr);
        RestartCount<TMatrix> precond_left_gmres_ir(
            &gen_lin_sys, args, precond_left_args
        );
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);
        EXPECT_LT(
            precond_left_gmres_ir.get_iteration(),
            gmres_ir.get_iteration()
        );

        PrecondArgPkg<TMatrix, double> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, double>>(),
            ilu_ptr
        );
        RestartCount<TMatrix> precond_right_gmres_ir(
            &gen_lin_sys, args, precond_right_args
        );
        precond_right_gmres_ir.solve();

        if (*show_plots) { precond_right_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_right_gmres_ir.check_converged());
        EXPECT_LE(precond_right_gmres_ir.get_relres(), args.target_rel_res);
        EXPECT_LT(
            precond_right_gmres_ir.get_iteration(),
            gmres_ir.get_iteration()
        );

    }

};

TEST_F(Preconditioned_MP_GMRES_IR_Test, SolveMinimumIterTest_SOLVER) {
    
    PreconditionedMinimumIterTest<MatrixDense>(solve_args);
    PreconditionedMinimumIterTest<NoFillMatrixSparse>(solve_args);

}

TEST_F(Preconditioned_MP_GMRES_IR_Test, SolveConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    PreconditionedSolveTest<MatrixDense>(A_path, b_path, solve_args);
    PreconditionedSolveTest<NoFillMatrixSparse>(A_path, b_path, solve_args);

}

TEST_F(Preconditioned_MP_GMRES_IR_Test, SolveConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense>(A_path, b_path, solve_args);
    PreconditionedSolveTest<NoFillMatrixSparse>(A_path, b_path, solve_args);

}

#include "test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Preconditioned_MP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg solve_args = SolveArgPkg(80, 10, Tol<double>::krylov_conv_tol());

    template <template <typename> typename M>
    void PreconditionedMinimumIterTest(
        const SolveArgPkg &args
    ) {

        fs::path A_path(solve_matrix_dir / fs::path("A_inv_45.csv"));
        fs::path Asqr_path(solve_matrix_dir / fs::path("Asqr_inv_45.csv"));
        fs::path Ainv_path(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
        fs::path b_path(solve_matrix_dir / fs::path("b_inv_45.csv"));

        M<double> A = read_matrixCSV<M, double>(TestBase::bundle, A_path);
        M<double> Asqr = read_matrixCSV<M, double>(TestBase::bundle, Asqr_path);
        Vector<double> b = read_matrixCSV<Vector, double>(TestBase::bundle, b_path);
        M<double> Ainv = read_matrixCSV<M, double>(TestBase::bundle, Ainv_path);

        GenericLinearSystem<M> gen_lin_sys(A, b);

        PrecondArgPkg<M, double> precond_left_args(
            std::make_shared<MatrixInversePreconditioner<M, double>>(Ainv)
        );
        SimpleConstantThreshold<M> precond_left_gmres_ir(&gen_lin_sys, args, precond_left_args);
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_left_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<M, double> precond_right_args(
            std::make_shared<NoPreconditioner<M, double>>(),
            std::make_shared<MatrixInversePreconditioner<M, double>>(Ainv)
        );
        SimpleConstantThreshold<M> precond_right_gmres_ir(&gen_lin_sys, args, precond_right_args);
        precond_right_gmres_ir.solve();

        if (*show_plots) { precond_right_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_right_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_right_gmres_ir.check_converged());
        EXPECT_LE(precond_right_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<M, double> precond_symmetric_args(
            std::make_shared<MatrixInversePreconditioner<M, double>>(Ainv),
            std::make_shared<MatrixInversePreconditioner<M, double>>(Ainv)
        );
        GenericLinearSystem<M> gen_lin_sys_Asqr(Asqr, b);

        SimpleConstantThreshold<M> precond_symmetric_gmres_ir(&gen_lin_sys_Asqr, args, precond_symmetric_args);
        precond_symmetric_gmres_ir.solve();

        if (*show_plots) { precond_symmetric_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_symmetric_gmres_ir.get_iteration(), 3);
        EXPECT_TRUE(precond_symmetric_gmres_ir.check_converged());
        EXPECT_LE(precond_symmetric_gmres_ir.get_relres(), args.target_rel_res);

    }

    template <template <typename> typename M>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args
    ) {

        M<double> A = read_matrixCSV<M, double>(TestBase::bundle, A_file_path);
        Vector<double> b = read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path);

        std::shared_ptr<ILUPreconditioner<M, double>> ilu_ptr(new ILUPreconditioner<M, double>(A));

        GenericLinearSystem<M> gen_lin_sys(A, b);

        RestartCount<M> gmres_ir(&gen_lin_sys, args);
        gmres_ir.solve();

        PrecondArgPkg<M, double> precond_left_args(ilu_ptr);
        RestartCount<M> precond_left_gmres_ir(&gen_lin_sys, args, precond_left_args);
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_left_gmres_ir.get_iteration(), gmres_ir.get_iteration());

        PrecondArgPkg<M, double> precond_right_args(
            std::make_shared<NoPreconditioner<M, double>>(),
            ilu_ptr
        );
        RestartCount<M> precond_right_gmres_ir(&gen_lin_sys, args, precond_right_args);
        precond_right_gmres_ir.solve();

        if (*show_plots) { precond_right_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_right_gmres_ir.check_converged());
        EXPECT_LE(precond_right_gmres_ir.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_right_gmres_ir.get_iteration(), gmres_ir.get_iteration());

    }

};

TEST_F(Preconditioned_MP_GMRES_IR_Test, MinimumIterTest_SOLVER) {
    
    PreconditionedMinimumIterTest<MatrixDense>(solve_args);
    PreconditionedMinimumIterTest<NoFillMatrixSparse>(solve_args);

}

TEST_F(Preconditioned_MP_GMRES_IR_Test, ConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    PreconditionedSolveTest<MatrixDense>(A_path, b_path, solve_args);
    PreconditionedSolveTest<NoFillMatrixSparse>(A_path, b_path, solve_args);

}

TEST_F(Preconditioned_MP_GMRES_IR_Test, ConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense>(A_path, b_path, solve_args);
    PreconditionedSolveTest<NoFillMatrixSparse>(A_path, b_path, solve_args);

}

#include "test.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class Preconditioned_FP_GMRES_IR_Test: public TestBase
{
public:

    const SolveArgPkg dbl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<double>::krylov_conv_tol()
    );
    const SolveArgPkg sgl_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<float>::krylov_conv_tol()
    );
    const SolveArgPkg hlf_GMRES_IR_args = SolveArgPkg(
        80, 10, Tol<__half>::krylov_conv_tol()
    );

    template <template <typename> typename TMatrix, typename TPrecision>
    void PreconditionedSingleIterTest(
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
            TestBase::bundle, Asqr_path);
        Vector<double> b = read_matrixCSV<Vector, double>(
            TestBase::bundle, b_path
        );
        TMatrix<double> Ainv = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, Ainv_path
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        PrecondArgPkg<TMatrix, TPrecision> precond_left_args(
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            )
        );
        FP_GMRES_IR_Solve<TMatrix, TPrecision> precond_left_gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(),
            args, precond_left_args
        );
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_left_gmres_ir.get_iteration(), 1);
        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<TMatrix, TPrecision> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>(),
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            )
        );
        FP_GMRES_IR_Solve<TMatrix, TPrecision> precond_right_gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(),
            args, precond_right_args
        );
        precond_right_gmres_ir.solve();

        if (*show_plots) { precond_right_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_right_gmres_ir.get_iteration(), 1);
        EXPECT_TRUE(precond_right_gmres_ir.check_converged());
        EXPECT_LE(precond_right_gmres_ir.get_relres(), args.target_rel_res);

        PrecondArgPkg<TMatrix, TPrecision> precond_symmetric_args(
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            )
        );
        GenericLinearSystem<TMatrix> gen_lin_sys_Asqr(Asqr, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys_Asqr(
            &gen_lin_sys_Asqr
        );

        FP_GMRES_IR_Solve<TMatrix, TPrecision> precond_symmetric_gmres_ir(
            &typed_lin_sys_Asqr, Tol<TPrecision>::roundoff(),
            args, precond_symmetric_args
        );
        precond_symmetric_gmres_ir.solve();

        if (*show_plots) { precond_symmetric_gmres_ir.view_relres_plot("log"); }

        EXPECT_EQ(precond_symmetric_gmres_ir.get_iteration(), 1);
        EXPECT_TRUE(precond_symmetric_gmres_ir.check_converged());
        EXPECT_LE(precond_symmetric_gmres_ir.get_relres(), args.target_rel_res);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args
    ) {

        TMatrix<double> A = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        );
        Vector<double> b = read_matrixCSV<Vector, double>(
            TestBase::bundle, b_file_path
        );

        std::shared_ptr<ILUPreconditioner<TMatrix, TPrecision>> ilu_ptr(
            new ILUPreconditioner<TMatrix, TPrecision>(
                A.template cast<TPrecision>()
            )
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        FP_GMRES_IR_Solve<TMatrix, TPrecision> gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(), args
        );
        gmres_ir.solve();

        PrecondArgPkg<TMatrix, TPrecision> precond_left_args(ilu_ptr);
        FP_GMRES_IR_Solve<TMatrix, TPrecision> precond_left_gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(),
            args, precond_left_args
        );
        precond_left_gmres_ir.solve();

        if (*show_plots) { precond_left_gmres_ir.view_relres_plot("log"); }

        EXPECT_TRUE(precond_left_gmres_ir.check_converged());
        EXPECT_LE(precond_left_gmres_ir.get_relres(), args.target_rel_res);
        EXPECT_LT(
            precond_left_gmres_ir.get_iteration(),
            gmres_ir.get_iteration()
        );

        PrecondArgPkg<TMatrix, TPrecision> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>(),
            ilu_ptr
        );
        FP_GMRES_IR_Solve<TMatrix, TPrecision> precond_right_gmres_ir(
            &typed_lin_sys, Tol<TPrecision>::roundoff(),
            args, precond_right_args
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

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveSingleIterTest_Double_SOLVER) {
    
    PreconditionedSingleIterTest<MatrixDense, double>(dbl_GMRES_IR_args);
    PreconditionedSingleIterTest<NoFillMatrixSparse, double>(dbl_GMRES_IR_args);

}

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveConvDiff256_Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));
    
    PreconditionedSolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args
    );

}

TEST_F(
    Preconditioned_FP_GMRES_IR_Test,
    SolveConvDiff1024_Double_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, double>(
        A_path, b_path, dbl_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, double>(
        A_path, b_path, dbl_GMRES_IR_args
    );

}

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveSingleIterTest_Single_SOLVER) {
    
    PreconditionedSingleIterTest<MatrixDense, float>(sgl_GMRES_IR_args);
    PreconditionedSingleIterTest<NoFillMatrixSparse, float>(sgl_GMRES_IR_args);

}

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveConvDiff256_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args
    );

}

TEST_F(
    Preconditioned_FP_GMRES_IR_Test,
    SolveConvDiff1024_Single_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(
        A_path, b_path, sgl_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, float>(
        A_path, b_path, sgl_GMRES_IR_args
    );

}

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveSingleIterTest_Half_SOLVER) {
    
    PreconditionedSingleIterTest<MatrixDense, __half>(hlf_GMRES_IR_args);
    PreconditionedSingleIterTest<NoFillMatrixSparse, __half>(hlf_GMRES_IR_args);

}

TEST_F(Preconditioned_FP_GMRES_IR_Test, SolveConvDiff256_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(
        A_path, b_path, hlf_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(
        A_path, b_path, hlf_GMRES_IR_args
    );

}

TEST_F(
    Preconditioned_FP_GMRES_IR_Test,
    HalfConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER
) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(
        A_path, b_path, hlf_GMRES_IR_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(
        A_path, b_path, hlf_GMRES_IR_args
    );

}
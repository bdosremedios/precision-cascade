#include "test.h"

#include "solvers/GMRES/GMRESSolve.h"

class PGMRES_Solve_Test: public TestBase
{
public:

    const int manual_n = 45;

    const SolveArgPkg dbl_manual_args = SolveArgPkg(
        manual_n, 10, Tol<double>::krylov_conv_tol()
    );
    const SolveArgPkg sgl_manual_args = SolveArgPkg(
        manual_n, 10, Tol<float>::krylov_conv_tol()
    );
    const SolveArgPkg hlf_manual_args = SolveArgPkg(
        manual_n, 10, Tol<half>::krylov_conv_tol()
    );

    const SolveArgPkg dbl_gen_args = SolveArgPkg(
        80, 10, Tol<double>::krylov_conv_tol()
    );
    const SolveArgPkg sgl_gen_args = SolveArgPkg(
        80, 10, Tol<float>::krylov_conv_tol()
    );
    const SolveArgPkg hlf_gen_args = SolveArgPkg(
        80, 10, Tol<half>::krylov_conv_tol()
    );

    void SetUp() {
        TestBase::SetUp();
    }

    template <template <typename> typename TMatrix>
    void TestMatchIdentity() {

        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("A_inv_45.csv")
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, solve_matrix_dir / fs::path("b_inv_45.csv")
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, double> typed_lin_sys(&gen_lin_sys);

        GMRESSolve<TMatrix, double> pgmres_solve_default(
            &typed_lin_sys, dbl_manual_args
        );

        PrecondArgPkg<TMatrix, double> noprecond(
            std::make_shared<NoPreconditioner<TMatrix, double>>(),
            std::make_shared<NoPreconditioner<TMatrix, double>>()
        );
        GMRESSolve<TMatrix, double> pgmres_solve_explicit_noprecond(
            &typed_lin_sys, dbl_manual_args, noprecond
        );

        PrecondArgPkg<TMatrix, double> identity(
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                TMatrix<double>::Identity(TestBase::bundle, manual_n, manual_n)
            ),
            std::make_shared<MatrixInversePreconditioner<TMatrix, double>>(
                TMatrix<double>::Identity(TestBase::bundle, manual_n, manual_n)
            )
        );
        GMRESSolve<TMatrix, double> pgmres_solve_inverse_of_identity(
            &typed_lin_sys, dbl_manual_args, identity
        );

        pgmres_solve_default.solve();
        if (*show_plots) {
            pgmres_solve_default.view_relres_plot("log");
        }
        pgmres_solve_explicit_noprecond.solve();
        if (*show_plots) {
            pgmres_solve_explicit_noprecond.view_relres_plot("log");
        }
        pgmres_solve_inverse_of_identity.solve();
        if (*show_plots) {
            pgmres_solve_inverse_of_identity.view_relres_plot("log");
        }

        double error_coeff = abs_ns::abs(
            (MatrixDense<double>(A).norm() *
             pgmres_solve_default.get_typed_soln().norm()
            ).get_scalar()
        );
        ASSERT_VECTOR_NEAR(
            pgmres_solve_default.get_typed_soln(),
            pgmres_solve_explicit_noprecond.get_typed_soln(),
            error_coeff*Tol<double>::gamma(manual_n)
        );
        ASSERT_VECTOR_NEAR(
            pgmres_solve_explicit_noprecond.get_typed_soln(),
            pgmres_solve_inverse_of_identity.get_typed_soln(),
            error_coeff*Tol<double>::gamma(manual_n)
        );
        ASSERT_VECTOR_NEAR(
            pgmres_solve_inverse_of_identity.get_typed_soln(),
            pgmres_solve_default.get_typed_soln(),
            error_coeff*Tol<double>::gamma(manual_n)
        );

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestPrecondSingleIter(
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
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        PrecondArgPkg<TMatrix, TPrecision> precond_left_args(
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            )
        );
        GMRESSolve<TMatrix, TPrecision> precond_left_gmres(
            &typed_lin_sys, args, precond_left_args
        );
        precond_left_gmres.solve();

        if (*show_plots) { precond_left_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_left_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_left_gmres.check_converged());
        EXPECT_LE(precond_left_gmres.get_relres(), args.target_rel_res);

        PrecondArgPkg<TMatrix, TPrecision> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>(),
            std::make_shared<MatrixInversePreconditioner<TMatrix, TPrecision>>(
                Ainv.template cast<TPrecision>()
            )
        );
        GMRESSolve<TMatrix, TPrecision> precond_right_gmres(
            &typed_lin_sys, args, precond_right_args
        );
        precond_right_gmres.solve();

        if (*show_plots) { precond_right_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_right_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_right_gmres.check_converged());
        EXPECT_LE(precond_right_gmres.get_relres(), args.target_rel_res);

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

        GMRESSolve<TMatrix, TPrecision> precond_symmetric_gmres(
            &typed_lin_sys_Asqr, args, precond_symmetric_args
        );
        precond_symmetric_gmres.solve();

        if (*show_plots) { precond_symmetric_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_symmetric_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_symmetric_gmres.check_converged());
        EXPECT_LE(precond_symmetric_gmres.get_relres(), args.target_rel_res);
    
    }   

    template <template <typename> typename TMatrix, typename TPrecision>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &Asqr_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args
    ) {

        TMatrix<double> A = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, A_file_path
        );
        TMatrix<double> A_sqr = read_matrixCSV<TMatrix, double>(
            TestBase::bundle, Asqr_file_path
        );
        Vector<double> b = read_vectorCSV<double>(
            TestBase::bundle, b_file_path
        );

        std::shared_ptr<ILUPreconditioner<TMatrix, TPrecision>> ilu_ptr = (
            std::make_shared<ILUPreconditioner<TMatrix, TPrecision>>(
                A.template cast<TPrecision>()
            )
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        GMRESSolve<TMatrix, TPrecision> gmres(
            &typed_lin_sys, args
        );
        gmres.solve();

        PrecondArgPkg<TMatrix, TPrecision> precond_left_args(ilu_ptr);
        GMRESSolve<TMatrix, TPrecision> precond_left_gmres(
            &typed_lin_sys, args, precond_left_args
        );
        precond_left_gmres.solve();

        if (*show_plots) { precond_left_gmres.view_relres_plot("log"); }

        EXPECT_TRUE(precond_left_gmres.check_converged());
        EXPECT_LE(precond_left_gmres.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_left_gmres.get_iteration(), gmres.get_iteration());

        PrecondArgPkg<TMatrix, TPrecision> precond_right_args(
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>(), ilu_ptr
        );
        GMRESSolve<TMatrix, TPrecision> precond_right_gmres(
            &typed_lin_sys, args, precond_right_args
        );
        precond_right_gmres.solve();

        if (*show_plots) { precond_right_gmres.view_relres_plot("log"); }

        EXPECT_TRUE(precond_right_gmres.check_converged());
        EXPECT_LE(precond_right_gmres.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_right_gmres.get_iteration(), gmres.get_iteration());

    }

};

TEST_F(PGMRES_Solve_Test, TestDefAndNoPreconditioningMatchesIdentity_SOLVER) {
    TestMatchIdentity<MatrixDense>();
    TestMatchIdentity<NoFillMatrixSparse>();
}

TEST_F(PGMRES_Solve_Test, SolveSingleIter_Double_SOLVER) {
    TestPrecondSingleIter<MatrixDense, double>(dbl_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, double>(dbl_manual_args);
}

TEST_F(PGMRES_Solve_Test, SolveConvDiff256_Double_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, double>(
        A_path, Asqr_path, b_path, dbl_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, double>(
        A_path, Asqr_path, b_path, dbl_gen_args
    );

}

TEST_F(PGMRES_Solve_Test, SolveConvDiff1024_Double_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, double>(
        A_path, Asqr_path, b_path, dbl_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, double>(
        A_path, Asqr_path, b_path, dbl_gen_args
    );

}

TEST_F(PGMRES_Solve_Test, SolveSingleIter_Single_SOLVER) {
    TestPrecondSingleIter<MatrixDense, float>(sgl_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, float>(sgl_manual_args);
}

TEST_F(PGMRES_Solve_Test, SolveConvDiff256_Single_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(
        A_path, Asqr_path, b_path, sgl_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, float>(
        A_path, Asqr_path, b_path, sgl_gen_args
    );

}

TEST_F(PGMRES_Solve_Test, SolveConvDiff1024_Single_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(
        A_path, Asqr_path, b_path, sgl_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, float>(
        A_path, Asqr_path, b_path, sgl_gen_args
    );

}

TEST_F(PGMRES_Solve_Test, SolveSingleIter_Half_SOLVER) {
    TestPrecondSingleIter<MatrixDense, __half>(hlf_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, __half>(hlf_manual_args);
}

TEST_F(PGMRES_Solve_Test, SolveConvDiff256_Half_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(
        A_path, Asqr_path, b_path, hlf_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(
        A_path, Asqr_path, b_path, hlf_gen_args
    );

}

TEST_F(PGMRES_Solve_Test, SolveConvDiff1024_Half_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(
        A_path, Asqr_path, b_path, hlf_gen_args
    );
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(
        A_path, Asqr_path, b_path, hlf_gen_args
    );

}
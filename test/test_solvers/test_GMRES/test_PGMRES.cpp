#include "../../test.h"

#include "solvers/GMRES/GMRESSolve.h"

class PGMRES_Solve_Test: public TestBase
{
public:

    const int manual_n = 45;

    const SolveArgPkg dbl_manual_args = SolveArgPkg(manual_n, 10, Tol<double>::krylov_conv_tol());
    const SolveArgPkg sgl_manual_args = SolveArgPkg(manual_n, 10, Tol<float>::krylov_conv_tol());
    const SolveArgPkg hlf_manual_args = SolveArgPkg(manual_n, 10, Tol<half>::krylov_conv_tol());

    const SolveArgPkg dbl_gen_args = SolveArgPkg(80, 10, Tol<double>::krylov_conv_tol());
    const SolveArgPkg sgl_gen_args = SolveArgPkg(80, 10, Tol<float>::krylov_conv_tol());
    const SolveArgPkg hlf_gen_args = SolveArgPkg(80, 10, Tol<half>::krylov_conv_tol());

    void SetUp() {
        TestBase::SetUp();
    }

    template <template <typename> typename M>
    void TestMatchIdentity() {

        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_inv_45.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_inv_45.csv"))
        );

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, double> typed_lin_sys(&gen_lin_sys);

        GMRESSolve<M, double> pgmres_solve_default(&typed_lin_sys, Tol<double>::roundoff(), dbl_manual_args);

        PrecondArgPkg<M, double> noprecond(
            std::make_shared<NoPreconditioner<M, double>>(),
            std::make_shared<NoPreconditioner<M, double>>()
        );
        GMRESSolve<M, double> pgmres_solve_explicit_noprecond(
            &typed_lin_sys, Tol<double>::roundoff(), dbl_manual_args, noprecond
        );

        PrecondArgPkg<M, double> identity(
            std::make_shared<MatrixInversePreconditioner<M, double>>(
                M<double>::Identity(TestBase::bundle, manual_n, manual_n)
            ),
            std::make_shared<MatrixInversePreconditioner<M, double>>(
                M<double>::Identity(TestBase::bundle, manual_n, manual_n)
            )
        );
        GMRESSolve<M, double> pgmres_solve_inverse_of_identity(
            &typed_lin_sys, Tol<double>::roundoff(), dbl_manual_args, identity
        );

        pgmres_solve_default.solve();
        if (*show_plots) { pgmres_solve_default.view_relres_plot("log"); }
        pgmres_solve_explicit_noprecond.solve();
        if (*show_plots) { pgmres_solve_explicit_noprecond.view_relres_plot("log"); }
        pgmres_solve_inverse_of_identity.solve();
        if (*show_plots) { pgmres_solve_inverse_of_identity.view_relres_plot("log"); }

        double error_coeff = std::abs(
            (MatrixDense<double>(A).norm()*pgmres_solve_default.get_typed_soln().norm()).get_scalar()
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

    template <template <typename> typename M, typename T>
    void TestPrecondSingleIter(
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
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        PrecondArgPkg<M, T> precond_left_args(
            std::make_shared<MatrixInversePreconditioner<M, T>>(Ainv.template cast<T>())
        );
        GMRESSolve<M, T> precond_left_gmres(
            &typed_lin_sys, Tol<T>::roundoff(), args, precond_left_args
        );
        precond_left_gmres.solve();

        if (*show_plots) { precond_left_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_left_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_left_gmres.check_converged());
        EXPECT_LE(precond_left_gmres.get_relres(), args.target_rel_res);

        PrecondArgPkg<M, T> precond_right_args(
            std::make_shared<NoPreconditioner<M, T>>(),
            std::make_shared<MatrixInversePreconditioner<M, T>>(Ainv.template cast<T>())
        );
        GMRESSolve<M, T> precond_right_gmres(
            &typed_lin_sys, Tol<T>::roundoff(), args, precond_right_args
        );
        precond_right_gmres.solve();

        if (*show_plots) { precond_right_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_right_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_right_gmres.check_converged());
        EXPECT_LE(precond_right_gmres.get_relres(), args.target_rel_res);

        PrecondArgPkg<M, T> precond_symmetric_args(
            std::make_shared<MatrixInversePreconditioner<M, T>>(Ainv.template cast<T>()),
            std::make_shared<MatrixInversePreconditioner<M, T>>(Ainv.template cast<T>())
        );
        GenericLinearSystem<M> gen_lin_sys_Asqr(Asqr, b);
        TypedLinearSystem<M, T> typed_lin_sys_Asqr(&gen_lin_sys_Asqr);

        GMRESSolve<M, T> precond_symmetric_gmres(
            &typed_lin_sys_Asqr, Tol<T>::roundoff(), args, precond_symmetric_args
        );
        precond_symmetric_gmres.solve();

        if (*show_plots) { precond_symmetric_gmres.view_relres_plot("log"); }

        EXPECT_EQ(precond_symmetric_gmres.get_iteration(), 1);
        EXPECT_TRUE(precond_symmetric_gmres.check_converged());
        EXPECT_LE(precond_symmetric_gmres.get_relres(), args.target_rel_res);
    
    }   

    template <template <typename> typename M, typename T>
    void PreconditionedSolveTest(
        const fs::path &A_file_path,
        const fs::path &Asqr_file_path,
        const fs::path &b_file_path,
        const SolveArgPkg &args
    ) {

        M<double> A = read_matrixCSV<M, double>(TestBase::bundle, A_file_path);
        M<double> A_sqr = read_matrixCSV<M, double>(TestBase::bundle, Asqr_file_path);
        Vector<double> b = read_matrixCSV<Vector, double>(TestBase::bundle, b_file_path);

        std::shared_ptr<ILUPreconditioner<M, T>> ilu_ptr = std::make_shared<ILUPreconditioner<M, T>>(
            A.template cast<T>()
        );

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        GMRESSolve<M, T> gmres(&typed_lin_sys, Tol<T>::roundoff(), args);
        gmres.solve();

        PrecondArgPkg<M, T> precond_left_args(ilu_ptr);
        GMRESSolve<M, T> precond_left_gmres(
            &typed_lin_sys, Tol<T>::roundoff(), args, precond_left_args
        );
        precond_left_gmres.solve();

        if (*show_plots) { precond_left_gmres.view_relres_plot("log"); }

        EXPECT_TRUE(precond_left_gmres.check_converged());
        EXPECT_LE(precond_left_gmres.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_left_gmres.get_iteration(), gmres.get_iteration());

        PrecondArgPkg<M, T> precond_right_args(std::make_shared<NoPreconditioner<M, T>>(), ilu_ptr);
        GMRESSolve<M, T> precond_right_gmres(
            &typed_lin_sys, Tol<T>::roundoff(), args, precond_right_args
        );
        precond_right_gmres.solve();

        if (*show_plots) { precond_right_gmres.view_relres_plot("log"); }

        EXPECT_TRUE(precond_right_gmres.check_converged());
        EXPECT_LE(precond_right_gmres.get_relres(), args.target_rel_res);
        EXPECT_LT(precond_right_gmres.get_iteration(), gmres.get_iteration());

    }

};

TEST_F(PGMRES_Solve_Test, TestDefaultandNoPreconditioningMatchesIdentity_SOLVER) {
    TestMatchIdentity<MatrixDense>();
    TestMatchIdentity<NoFillMatrixSparse>();
}

TEST_F(PGMRES_Solve_Test, TestPrecondSingleIter_Double) {
    TestPrecondSingleIter<MatrixDense, double>(dbl_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, double>(dbl_manual_args);
}

TEST_F(PGMRES_Solve_Test, DoubleConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, double>(A_path, Asqr_path, b_path, dbl_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, double>(A_path, Asqr_path, b_path, dbl_gen_args);

}

TEST_F(PGMRES_Solve_Test, DoubleConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, double>(A_path, Asqr_path, b_path, dbl_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, double>(A_path, Asqr_path, b_path, dbl_gen_args);

}

TEST_F(PGMRES_Solve_Test, TestPrecondSingleIter_Single) {
    TestPrecondSingleIter<MatrixDense, float>(sgl_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, float>(sgl_manual_args);
}

TEST_F(PGMRES_Solve_Test, SingleConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(A_path, Asqr_path, b_path, sgl_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, float>(A_path, Asqr_path, b_path, sgl_gen_args);

}

TEST_F(PGMRES_Solve_Test, SingleConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, float>(A_path, Asqr_path, b_path, sgl_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, float>(A_path, Asqr_path, b_path, sgl_gen_args);

}

TEST_F(PGMRES_Solve_Test, TestPrecondSingleIter_Half) {
    TestPrecondSingleIter<MatrixDense, __half>(hlf_manual_args);
    TestPrecondSingleIter<NoFillMatrixSparse, __half>(hlf_manual_args);
}

TEST_F(PGMRES_Solve_Test, HalfConvergenceTest_ConvDiff256_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_256_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_256_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_256_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(A_path, Asqr_path, b_path, hlf_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(A_path, Asqr_path, b_path, hlf_gen_args);

}

TEST_F(PGMRES_Solve_Test, HalfConvergenceTest_ConvDiff1024_LONGRUNTIME_SOLVER) {

    fs::path A_path(solve_matrix_dir / fs::path("conv_diff_1024_A.csv"));
    fs::path Asqr_path(solve_matrix_dir / fs::path("conv_diff_1024_Asqr.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("conv_diff_1024_b.csv"));

    PreconditionedSolveTest<MatrixDense, __half>(A_path, Asqr_path, b_path, hlf_gen_args);
    PreconditionedSolveTest<NoFillMatrixSparse, __half>(A_path, Asqr_path, b_path, hlf_gen_args);

}
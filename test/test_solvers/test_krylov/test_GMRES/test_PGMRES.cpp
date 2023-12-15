#include "../../../test.h"

#include "solvers/krylov/GMRES.h"

class PGMRES_Solve_Test: public TestBase
{
public:

    SolveArgPkg pgmres_args;

    void SetUp() { pgmres_args.target_rel_res = conv_tol_dbl; }

    template <template <typename> typename M>
    void TestMatchIdentity() {

        constexpr int n(45);
        M<double> A(read_matrixCSV<M, double>(solve_matrix_dir / fs::path("A_inv_45.csv")));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv")));
        TypedLinearSystem<M, double> lin_sys(A, b);

        GMRESSolve<M, double> pgmres_solve_default(lin_sys, Tol<double>::roundoff(), pgmres_args);

        PrecondArgPkg<M, double> noprecond(make_shared<NoPreconditioner<M, double>>(),
                                           make_shared<NoPreconditioner<M, double>>());
        GMRESSolve<M, double> pgmres_solve_explicit_noprecond(lin_sys, Tol<double>::roundoff(), pgmres_args, noprecond);

        PrecondArgPkg<M, double> identity(make_shared<MatrixInverse<M, double>>(M<double>::Identity(n, n)),
                                          make_shared<MatrixInverse<M, double>>(M<double>::Identity(n, n)));
        GMRESSolve<M, double> pgmres_solve_inverse_of_identity(lin_sys, Tol<double>::roundoff(), pgmres_args, identity);

        pgmres_solve_default.solve();
        if (*show_plots) { pgmres_solve_default.view_relres_plot("log"); }
        pgmres_solve_explicit_noprecond.solve();
        if (*show_plots) { pgmres_solve_explicit_noprecond.view_relres_plot("log"); }
        pgmres_solve_inverse_of_identity.solve();
        if (*show_plots) { pgmres_solve_inverse_of_identity.view_relres_plot("log"); }

        ASSERT_VECTOR_EQ(pgmres_solve_default.get_typed_soln(),
                         pgmres_solve_explicit_noprecond.get_typed_soln());
        ASSERT_VECTOR_EQ(pgmres_solve_explicit_noprecond.get_typed_soln(),
                         pgmres_solve_inverse_of_identity.get_typed_soln());
        ASSERT_VECTOR_EQ(pgmres_solve_inverse_of_identity.get_typed_soln(),
                         pgmres_solve_default.get_typed_soln());

    }

    template <template <typename> typename M>
    void TestPrecondSingleIter(
        const M<double> &A,
        const MatrixVector<double> &b,
        const PrecondArgPkg<M, double> &precond_pkg
    ) {

        TypedLinearSystem<M, double> lin_sys(A, b);
        GMRESSolve<M, double> pgmres_solve(lin_sys, Tol<double>::roundoff(), pgmres_args, precond_pkg);

        pgmres_solve.solve();
        if (*show_plots) { pgmres_solve.view_relres_plot("log"); }

        EXPECT_EQ(pgmres_solve.get_iteration(), 1);
        EXPECT_TRUE(pgmres_solve.check_converged());
        EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);
    }

    template <template <typename> typename M>
    void TestPrecond3IterAndMatch(
        const M<double> &A,
        const MatrixVector<double> &b,
        const MatrixVector<double> &x_test,
        const PrecondArgPkg<M, double> &precond_pkg
    ) {

        TypedLinearSystem<M, double> lin_sys(A, b);
        GMRESSolve<M, double> pgmres_solve(lin_sys, Tol<double>::roundoff(), pgmres_args, precond_pkg);

        pgmres_solve.solve();
        if (*show_plots) { pgmres_solve.view_relres_plot("log"); }

        EXPECT_EQ(pgmres_solve.get_iteration(), 3);
        EXPECT_TRUE(pgmres_solve.check_converged());
        EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);
    
        EXPECT_LE((pgmres_solve.get_typed_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

    }

    template <template <typename> typename M>
    void TestLeftPreconditioning(fs::path A_path, fs::path Ainv_path, fs::path b_path) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));

        TestPrecondSingleIter<M>(
            A, b,
            PrecondArgPkg<M, double>(make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

    template <template <typename> typename M>
    void TestRightPreconditioning(fs::path A_path, fs::path Ainv_path, fs::path b_path) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));

        TestPrecondSingleIter<M>(
            A, b,
            PrecondArgPkg<M, double>(make_shared<NoPreconditioner<M, double>>(),
                                     make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

    template <template <typename> typename M>
    void TestSymmetricPreconditioning(fs::path A_path, fs::path Ainv_path, fs::path b_path) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));

        TestPrecondSingleIter<M>(
            A*A, b,
            PrecondArgPkg<M, double>(make_shared<MatrixInverse<M, double>>(Ainv),
                                     make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

    template <template <typename> typename M>
    void TestLeftPreconditioning_3eigs(
        fs::path A_path, fs::path Ainv_path, fs::path b_path, fs::path xtest_path
    ) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));
        MatrixVector<double> xtest(read_matrixCSV<MatrixVector, double>(xtest_path));

        TestPrecond3IterAndMatch<M>(
            A, b, xtest,
            PrecondArgPkg<M, double>(make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

    template <template <typename> typename M>
    void TestRightPreconditioning_3eigs(
        fs::path A_path, fs::path Ainv_path, fs::path b_path, fs::path xtest_path
    ) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));
        MatrixVector<double> xtest(read_matrixCSV<MatrixVector, double>(xtest_path));

        TestPrecond3IterAndMatch<M>(
            A, b, xtest,
            PrecondArgPkg<M, double>(make_shared<NoPreconditioner<M, double>>(),
                                     make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

    template <template <typename> typename M>
    void TestSymmetricPreconditioning_3eigs(
        fs::path A_path, fs::path Ainv_path, fs::path b_path, fs::path xtest_path
    ) {

        M<double> A(read_matrixCSV<M, double>(A_path));
        M<double> Ainv(read_matrixCSV<M, double>(Ainv_path));
        MatrixVector<double> b(read_matrixCSV<MatrixVector, double>(b_path));
        MatrixVector<double> xtest(read_matrixCSV<MatrixVector, double>(xtest_path));

        TestPrecond3IterAndMatch<M>(
            A*A, b, xtest,
            PrecondArgPkg<M, double>(make_shared<MatrixInverse<M, double>>(Ainv),
                                     make_shared<MatrixInverse<M, double>>(Ainv))
        );

    }

};

TEST_F(PGMRES_Solve_Test, TestDefaultandNoPreconditioningMatchesIdentity) {
    TestMatchIdentity<MatrixDense>();
    TestMatchIdentity<MatrixSparse>();
}

TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_RandA45) {

    fs::path A_path(solve_matrix_dir / fs::path("A_inv_45.csv"));
    fs::path Ainv_path(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_inv_45.csv"));

    TestLeftPreconditioning<MatrixDense>(A_path, Ainv_path, b_path);
    TestLeftPreconditioning<MatrixSparse>(A_path, Ainv_path, b_path);

}

TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_RandA45) {

    fs::path A_path(solve_matrix_dir / fs::path("A_inv_45.csv"));
    fs::path Ainv_path(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_inv_45.csv"));

    TestRightPreconditioning<MatrixDense>(A_path, Ainv_path, b_path);
    TestRightPreconditioning<MatrixSparse>(A_path, Ainv_path, b_path);

}

TEST_F(PGMRES_Solve_Test, TestSymmeticPreconditioning_RandA45) {

    fs::path A_path(solve_matrix_dir / fs::path("A_inv_45.csv"));
    fs::path Ainv_path(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_inv_45.csv"));

    TestSymmetricPreconditioning<MatrixDense>(A_path, Ainv_path, b_path);
    TestSymmetricPreconditioning<MatrixSparse>(A_path, Ainv_path, b_path);

}

TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_3eigs) {

    fs::path A_path(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    fs::path Ainv_path(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    fs::path xtest_path(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestLeftPreconditioning_3eigs<MatrixDense>(A_path, Ainv_path, b_path, xtest_path);
    TestLeftPreconditioning_3eigs<MatrixSparse>(A_path, Ainv_path, b_path, xtest_path);

}

TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_3eigs) {

    fs::path A_path(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    fs::path Ainv_path(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    fs::path b_path(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    fs::path xtest_path(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestRightPreconditioning_3eigs<MatrixDense>(A_path, Ainv_path, b_path, xtest_path);
    TestRightPreconditioning_3eigs<MatrixSparse>(A_path, Ainv_path, b_path, xtest_path);

}
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
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
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

};

TEST_F(PGMRES_Solve_Test, TestDefaultandNoPreconditioningMatchesIdentity_Dense) {
    TestMatchIdentity<MatrixDense>();
}
TEST_F(PGMRES_Solve_Test, TestDefaultandNoPreconditioningMatchesIdentity_Sparse) {
    TestMatchIdentity<MatrixSparse>();
}

TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_RandA45_Dense) {
    constexpr int n(45);
    MatrixDense<double> A = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixDense<double> Ainv = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixDense>(
        A, b,
        PrecondArgPkg<MatrixDense, double>(make_shared<MatrixInverse<MatrixDense, double>>(Ainv))
    );
}
TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_RandA45_Sparse) {
    constexpr int n(45);
    MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixSparse<double> Ainv = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixSparse>(
        A, b,
        PrecondArgPkg<MatrixSparse, double>(make_shared<MatrixInverse<MatrixSparse, double>>(Ainv))
    );
}


TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_RandA45_Dense) {
    constexpr int n(45);
    MatrixDense<double> A = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixDense<double> Ainv = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixDense>(
        A, b,
        PrecondArgPkg<MatrixDense, double>(make_shared<NoPreconditioner<MatrixDense, double>>(),
                                           make_shared<MatrixInverse<MatrixDense, double>>(Ainv))
    );
}
TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_RandA45_Sparse) {
    constexpr int n(45);
    MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixSparse<double> Ainv = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixSparse>(
        A, b,
        PrecondArgPkg<MatrixSparse, double>(make_shared<NoPreconditioner<MatrixSparse, double>>(),
                                            make_shared<MatrixInverse<MatrixSparse, double>>(Ainv))
    );
}

TEST_F(PGMRES_Solve_Test, TestSymmeticPreconditioning_RandA45_Dense) {
    constexpr int n(45);
    MatrixDense<double> A = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixDense<double> Ainv = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixDense>(
        A*A, b,
        PrecondArgPkg<MatrixDense, double>(make_shared<MatrixInverse<MatrixDense, double>>(Ainv),
                                           make_shared<MatrixInverse<MatrixDense, double>>(Ainv))
    );
}
TEST_F(PGMRES_Solve_Test, TestSymmeticPreconditioning_RandA45_Sparse) {
    constexpr int n(45);
    MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_inv_45.csv"));
    MatrixSparse<double> Ainv = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("Ainv_inv_45.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_inv_45.csv"));
    TestPrecondSingleIter<MatrixSparse>(
        A*A, b,
        PrecondArgPkg<MatrixSparse, double>(make_shared<MatrixInverse<MatrixSparse, double>>(Ainv),
                                            make_shared<MatrixInverse<MatrixSparse, double>>(Ainv))
    );
}

TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_3eigs_Dense) {

    constexpr int n(25);
    MatrixDense<double> A = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    MatrixDense<double> Ainv = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    MatrixVector<double> x_test = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestPrecond3IterAndMatch<MatrixDense>(
        A, b, x_test,
        PrecondArgPkg<MatrixDense, double>(make_shared<MatrixInverse<MatrixDense, double>>(Ainv))
    );

}
TEST_F(PGMRES_Solve_Test, TestLeftPreconditioning_3eigs_Sparse) {

    constexpr int n(25);
    MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    MatrixSparse<double> Ainv = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    MatrixVector<double> x_test = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestPrecond3IterAndMatch<MatrixSparse>(
        A, b, x_test,
        PrecondArgPkg<MatrixSparse, double>(make_shared<MatrixInverse<MatrixSparse, double>>(Ainv))
    );

}

TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_3eigs_Dense) {

    constexpr int n(25);
    MatrixDense<double> A = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    MatrixDense<double> Ainv = read_matrixCSV<MatrixDense, double>(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    MatrixVector<double> x_test = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestPrecond3IterAndMatch<MatrixDense>(
        A, b, x_test,
        PrecondArgPkg<MatrixDense, double>(
            make_shared<NoPreconditioner<MatrixDense, double>>(),
            make_shared<MatrixInverse<MatrixDense, double>>(Ainv)
        )
    );

}
TEST_F(PGMRES_Solve_Test, TestRightPreconditioning_3eigs_Sparse) {

    constexpr int n(25);
    MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_25_saddle.csv"));
    MatrixSparse<double> Ainv = read_matrixCSV<MatrixSparse, double>(solve_matrix_dir / fs::path("A_25_invprecond_saddle.csv"));
    MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("b_25_saddle.csv"));
    MatrixVector<double> x_test = read_matrixCSV<MatrixVector, double>(solve_matrix_dir / fs::path("x_25_saddle.csv"));

    TestPrecond3IterAndMatch<MatrixSparse>(
        A, b, x_test,
        PrecondArgPkg<MatrixSparse, double>(
            make_shared<NoPreconditioner<MatrixSparse, double>>(),
            make_shared<MatrixInverse<MatrixSparse, double>>(Ainv)
        )
    );

}
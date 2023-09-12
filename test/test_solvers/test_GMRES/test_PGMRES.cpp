#include "../../test.h"

#include "solvers/krylov/GMRES.h"

class PGMRESSolveTest: public TestBase {


    public:
        
        SolveArgPkg pgmres_args;
    
        void SetUp() { pgmres_args.target_rel_res = conv_tol_dbl; }


};

TEST_F(PGMRESSolveTest, TestDefaultandNoPreconditioningMatchesIdentity) {
    
    constexpr int n(45);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));

    GMRESSolve<double> pgmres_solve_default(A, b, u_dbl, pgmres_args);
    GMRESSolve<double> pgmres_solve_explicit_noprecond(
        A, b, u_dbl,
        make_shared<NoPreconditioner<double>>(),
        make_shared<NoPreconditioner<double>>(),
        pgmres_args
    );
    GMRESSolve<double> pgmres_solve_inverse_of_identity(
        A, b, u_dbl,
        make_shared<MatrixInverse<double>>(Matrix<double, Dynamic, Dynamic>::Identity(n, n)),
        make_shared<MatrixInverse<double>>(Matrix<double, Dynamic, Dynamic>::Identity(n, n)),
        pgmres_args
    );

    pgmres_solve_default.solve();
    if (*show_plots) { pgmres_solve_default.view_relres_plot("log"); }
    pgmres_solve_explicit_noprecond.solve();
    if (*show_plots) { pgmres_solve_explicit_noprecond.view_relres_plot("log"); }
    pgmres_solve_inverse_of_identity.solve();
    if (*show_plots) { pgmres_solve_inverse_of_identity.view_relres_plot("log"); }

    EXPECT_EQ(pgmres_solve_default.get_typed_soln(), pgmres_solve_explicit_noprecond.get_typed_soln());
    EXPECT_EQ(pgmres_solve_explicit_noprecond.get_typed_soln(), pgmres_solve_inverse_of_identity.get_typed_soln());
    EXPECT_EQ(pgmres_solve_inverse_of_identity.get_typed_soln(), pgmres_solve_default.get_typed_soln());

}

TEST_F(PGMRESSolveTest, TestLeftPreconditioning_RandA45) {

    constexpr int n(45);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, n, n> Ainv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl, make_shared<MatrixInverse<double>>(Ainv), pgmres_args
    );

    pgmres_solve.solve();
    if (*show_plots) { pgmres_solve.view_relres_plot("log"); }
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 1);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

}

TEST_F(PGMRESSolveTest, TestRightPreconditioning_RandA45) {

    constexpr int n(45);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, n, n> Ainv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl,
        make_shared<NoPreconditioner<double>>(),
        make_shared<MatrixInverse<double>>(Ainv),
        pgmres_args
    );

    pgmres_solve.solve();
    if (*show_plots) { pgmres_solve.view_relres_plot("log"); }
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 1);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

}

TEST_F(PGMRESSolveTest, TestSymmetricPreconditioning_RandA45) {

    constexpr int n(45);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, n, n> Ainv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));
    GMRESSolve<double> pgmres_solve(
        A*A, b, u_dbl,
        make_shared<MatrixInverse<double>>(Ainv),
        make_shared<MatrixInverse<double>>(Ainv),
        pgmres_args
    );

    pgmres_solve.solve();
    if (*show_plots) { pgmres_solve.view_relres_plot("log"); }
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 1);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

}

TEST_F(PGMRESSolveTest, TestLeftPreconditioning_3eigs) {

    constexpr int n(25);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_saddle.csv"));
    Matrix<double, n, n> Ainv(read_matrix_csv<double>(solve_matrix_dir + "A_25_invprecond_saddle.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_saddle.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl,
        make_shared<MatrixInverse<double>>(Ainv),
        pgmres_args
    );

    pgmres_solve.solve();
    if (*show_plots) { pgmres_solve.view_relres_plot("log"); }
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 3);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_25_saddle.csv"));
    EXPECT_LE((pgmres_solve.get_typed_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(PGMRESSolveTest, TestRightPreconditioning_3eigs) {

    constexpr int n(25);
    Matrix<double, n, n> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_saddle.csv"));
    Matrix<double, n, n> Ainv(read_matrix_csv<double>(solve_matrix_dir + "A_25_invprecond_saddle.csv"));
    Matrix<double, n, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_saddle.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl,
        make_shared<NoPreconditioner<double>>(),
        make_shared<MatrixInverse<double>>(Ainv),
        pgmres_args
    );

    pgmres_solve.solve();
    if (*show_plots) { pgmres_solve.view_relres_plot("log"); }
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 3);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, n, 1> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_25_saddle.csv"));
    EXPECT_LE((pgmres_solve.get_typed_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}
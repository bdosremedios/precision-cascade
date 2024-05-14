#include "../../test.h"

#include "test_GMRESSolve.h"

class GMRESSolve_Component_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void CheckConstruction(const int &n) {

        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, n));
        TypedLinearSystem<M, double> lin_sys(A, b);

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), default_args);

        ASSERT_EQ(test_mock.max_kry_dim, n);
        ASSERT_NEAR(
            test_mock.rho.get_scalar(),
            (b - A*Vector<double>::Ones(TestBase::bundle, n)).norm().get_scalar(),
            std::abs((b - A*Vector<double>::Ones(TestBase::bundle, n)).norm().get_scalar())*
                Tol<double>::roundoff()
        );
        
        ASSERT_EQ(test_mock.Q_kry_basis.rows(), n);
        ASSERT_EQ(test_mock.Q_kry_basis.cols(), n);
        ASSERT_MATRIX_ZERO(test_mock.Q_kry_basis, 0.);

        ASSERT_EQ(test_mock.H_k.rows(), n+1);
        ASSERT_VECTOR_ZERO(test_mock.H_k, 0.);

        ASSERT_EQ(test_mock.H_Q.rows(), n+1);
        ASSERT_EQ(test_mock.H_Q.cols(), n+1);
        ASSERT_MATRIX_IDENTITY(test_mock.H_Q, 0.);

        ASSERT_EQ(test_mock.H_R.rows(), n+1);
        ASSERT_EQ(test_mock.H_R.cols(), n);
        ASSERT_MATRIX_ZERO(test_mock.H_R, 0.);

    }

    template <template <typename> typename M>
    void KrylovInitAndUpdate() {

        const double approx_cond_A_upbound(5.5);
        const int n(5);

        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_5_toy.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_5_toy.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), default_args);

        // Manually instantiate initial guess
        test_mock.typed_soln = Vector<double>::Ones(TestBase::bundle, n);
        Vector<double> r_0(b - A*Vector<double>::Ones(TestBase::bundle, n));

        // Create matrix to store previous basis vectors to ensure no change across iterations
        MatrixDense<double> Q_save(MatrixDense<double>::Zero(TestBase::bundle, n, n));
        Vector<double> H_k_save(Vector<double>::Zero(TestBase::bundle, n+1));

        // First update check first vector for basis is residual norm
        // and that Hessenberg first vector contructs next vector with entries
        test_mock.iterate_no_soln_solve();

        Vector<double> next_q(test_mock.Q_kry_basis.get_col(0));
        ASSERT_VECTOR_NEAR(
            next_q,
            r_0/r_0.norm(),
            Tol<double>::roundoff()
        );
        next_q = A*next_q;
        next_q -= test_mock.Q_kry_basis.get_col(0).copy_to_vec()*test_mock.H_k.get_elem(0);
        ASSERT_NEAR(
            next_q.norm().get_scalar(),
            test_mock.H_k.get_elem(1).get_scalar(),
            Tol<double>::roundoff()
        );
        ASSERT_VECTOR_NEAR(
            test_mock.next_q,
            next_q,
            Tol<double>::roundoff()
        );

        // Save basis to check that they remain unchanged
        Q_save.get_col(0).set_from_vec(test_mock.Q_kry_basis.get_col(0));

        // Subsequent updates
        for (int k=1; k<n; ++k) {

            // Iterate Krylov subspace and Hessenberg
            test_mock.iterate_no_soln_solve();
            Q_save.get_col(k).set_from_vec(test_mock.Q_kry_basis.get_col(k));

            // Get newly generated basis vector
            Vector<double> q(test_mock.Q_kry_basis.get_col(k));

            // Confirm that previous vectors are unchanged and are orthogonal to new one
            for (int j=0; j<k; ++j) {
                ASSERT_VECTOR_EQ(
                    test_mock.Q_kry_basis.get_col(j).copy_to_vec(),
                    Q_save.get_col(j).copy_to_vec()
                );
                ASSERT_NEAR(
                    test_mock.Q_kry_basis.get_col(j).copy_to_vec().dot(q).get_scalar(),
                    0.,
                    Tol<double>::loss_of_ortho_tol(approx_cond_A_upbound, k)
                );
            }

            // Confirm that Hessenberg matrix column corresponding to new basis vector
            // approximately constructs the next basis vector
            Vector<double> construct_q(test_mock.Q_kry_basis.get_col(k));
            construct_q = A*construct_q;
            for (int i=0; i<=k; ++i) {
                ASSERT_NEAR(
                    test_mock.Q_kry_basis.get_col(i).copy_to_vec().dot(construct_q).get_scalar(),
                    test_mock.H_k.get_elem(i).get_scalar(),
                    Tol<double>::roundoff()
                );
                construct_q -= test_mock.Q_kry_basis.get_col(i).copy_to_vec()*test_mock.H_k.get_elem(i);
            }
            ASSERT_NEAR(
                construct_q.norm().get_scalar(),
                test_mock.H_k.get_elem(k+1).get_scalar(),
                Tol<double>::roundoff()
            );

        }
    
    }

    template <template <typename> typename M>
    void H_QR_Update() {

        const double approx_cond_A_upbound(5.5);
        const int n(5);

        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_5_toy.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_5_toy.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), default_args);

        // Manually instantiate initial guess
        test_mock.typed_soln = Vector<double>::Ones(TestBase::bundle, n);
        Vector<double> r_0(b - A*Vector<double>::Ones(TestBase::bundle, n));

        MatrixDense<double> save_H_Q(MatrixDense<double>::Zero(TestBase::bundle, n+1, n+1));
        MatrixDense<double> save_H_R(MatrixDense<double>::Zero(TestBase::bundle, n+1, n));

        for (int kry_dim=1; kry_dim<=n; ++kry_dim) {

            int k = kry_dim-1;

            // Perform iteration and update QR_fact
            test_mock.iterate_no_soln_solve();
            test_mock.update_QR_fact();

            // Check that previous columns are unchanged by new update
            for (int i=0; i<k; ++i) {
                ASSERT_VECTOR_EQ(
                    test_mock.H_Q.get_col(i).copy_to_vec(),
                    save_H_Q.get_col(i).copy_to_vec()
                );
                ASSERT_VECTOR_EQ(
                    test_mock.H_R.get_col(i).copy_to_vec(),
                    save_H_R.get_col(i).copy_to_vec()
                );
            }

            // Save new columns generated
            save_H_Q.get_col(k).set_from_vec(test_mock.H_Q.get_col(k));
            save_H_R.get_col(k).set_from_vec(test_mock.H_R.get_col(k));

            // Test that k+1 by k+1 block of H_Q is orthogonal
            MatrixDense<double> H_Q_block(test_mock.H_Q.get_block(0, 0, k+2, k+2));
            MatrixDense<double> orthog_check(H_Q_block*H_Q_block.transpose());
            ASSERT_MATRIX_IDENTITY(
                orthog_check,
                Tol<double>::loss_of_ortho_tol(approx_cond_A_upbound, k+2)
            );

            // Test that k+1 by k block of H_R is uppertriangular
            ASSERT_MATRIX_UPPTRI(
                test_mock.H_R.get_block(0, 0, k+2, k+1).copy_to_mat(),
                Tol<double>::roundoff()
            );

        }

    }

    template <template <typename> typename M>
    void Update_x_Back_Substitution() {

        const double approx_R_cond_number_upbound(1.1);
        const int n(7);

        MatrixDense<double> Q(
            read_matrixCSV<MatrixDense, double>(TestBase::bundle, solve_matrix_dir / fs::path("Q_8_backsub.csv"))
        );
        MatrixDense<double> R(
            read_matrixCSV<MatrixDense, double>(TestBase::bundle, solve_matrix_dir / fs::path("R_8_backsub.csv"))
        );
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_7_dummy_backsub.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_7_dummy_backsub.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        // Set initial guess to zeros such that residual is just b
        Vector<double> x_0(Vector<double>::Zero(TestBase::bundle, n));
        SolveArgPkg args;
        args.init_guess = x_0;

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), args);

        // Set test_mock krylov basis to the identity to have typed_soln be directly the solved coefficients
        // of the back substitution
        test_mock.Q_kry_basis = MatrixDense<double>::Identity(TestBase::bundle, n, n);

        // Set premade Q R decomposition for H
        test_mock.H_Q = Q;
        test_mock.H_R = R;

        // Test that for each possible Hessenberg size determined by the Krylov subspace dimension
        // that the coefficient solution matches the pre-determined correct one from MATLAB solving
        for (int kry_dim=1; kry_dim<=n; ++kry_dim) {

            // Set Krylov subspace dim
            test_mock.curr_kry_dim = kry_dim;

            // Get relevant upper triangular system to solve
            MatrixDense<double> H_R_block(
                test_mock.H_R.get_block(0, 0, kry_dim, kry_dim).copy_to_mat()
            );
            
            // Load test solution
            Vector<double> test_soln(
                read_matrixCSV<Vector, double>(
                    TestBase::bundle, solve_matrix_dir / fs::path("x_" + std::to_string(kry_dim) + "_backsub.csv")
                )
            );

            // Solve with backsubstitution
            test_mock.update_x_minimizing_res();

            for (int i=0; i<kry_dim; ++i) {
                ASSERT_NEAR(
                    test_mock.typed_soln.get_elem(i).get_scalar(),
                    test_soln.get_elem(i).get_scalar(), 
                    2*mat_max_mag(test_soln)*Tol<double>::substitution_tol(approx_R_cond_number_upbound, n)
                );
            }

        }

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakFirstIter () {

        const int n(5);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_5_easysoln.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_5_easysoln.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        // Instantiate initial guess as true solution
        Vector<double> soln(Vector<double>::Ones(TestBase::bundle, n));
        SolveArgPkg args;
        args.target_rel_res = Tol<double>::krylov_conv_tol();
        args.init_guess = soln;

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), args);

        // Attempt to update subspace and Hessenberg
        test_mock.iterate();

        // Check basis Q and H are empty and Krylov hasn't been updated since already
        // hit the lucky break so can't build subspace and check that terminated but
        // not converged
        EXPECT_FALSE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.curr_kry_dim, 0);
        ASSERT_MATRIX_ZERO(test_mock.Q_kry_basis, Tol<double>::roundoff());
        ASSERT_VECTOR_ZERO(test_mock.H_k, Tol<double>::roundoff());

        // Attempt to solve and check that iteration does not occur since
        // should be terminated already but that convergence is updated
        test_mock.solve();
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_EQ(test_mock.get_iteration(), 0);

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakLaterIter() {

        constexpr int n(5);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_5_easysoln.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_5_easysoln.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        // Instantiate initial guess as true solution
        Vector<double> soln(Vector<double>::Zero(TestBase::bundle, n));
        soln.set_elem(0, SCALAR_ONE_D);
        SolveArgPkg args;
        args.init_guess = soln;

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), args);

        // Attempt to update subspace and convergence twice
        test_mock.iterate();
        test_mock.iterate();
        
        // Check basis Q has one normalized basis vector and others are empty and
        // has been marked terminated but not converged since we want to delay that
        // check for LinearSolve
        EXPECT_FALSE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.curr_kry_dim, 1);
        EXPECT_NEAR(
            test_mock.Q_kry_basis.get_col(0).copy_to_vec().norm().get_scalar(),
            1,
            Tol<double>::gamma(n)
        );
        ASSERT_MATRIX_ZERO(
            test_mock.Q_kry_basis.get_block(0, 1, n, n-1).copy_to_mat(),
            Tol<double>::roundoff()
        );

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakThroughSolve() {

        constexpr int n(5);
        M<double> A(
            read_matrixCSV<M, double>(TestBase::bundle, solve_matrix_dir / fs::path("A_5_easysoln.csv"))
        );
        Vector<double> b(
            read_matrixCSV<Vector, double>(TestBase::bundle, solve_matrix_dir / fs::path("b_5_easysoln.csv"))
        );
        TypedLinearSystem<M, double> lin_sys(A, b);

        // Instantiate initial guess as true solution
        Vector<double> soln(Vector<double>::Zero(TestBase::bundle, n));
        soln.set_elem(0, SCALAR_ONE_D);
        SolveArgPkg args;
        args.init_guess = soln;
        args.target_rel_res = Tol<double>::krylov_conv_tol();

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), args);

        // Attempt to update and solve through solve of LinearSolve
        test_mock.solve();
        
        // Check we have terminated at second iteration and have converged
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.get_iteration(), 1);

        // Check that subspace has not gone beyond 1 dimension and that krylov basis
        // as expected to have only a single column
        EXPECT_EQ(test_mock.curr_kry_dim, 1);
        EXPECT_NEAR(
            test_mock.Q_kry_basis.get_col(0).copy_to_vec().norm().get_scalar(),
            1,
            Tol<double>::gamma(n)
        );
        ASSERT_MATRIX_ZERO(
            test_mock.Q_kry_basis.get_block(0, 1, n, n-1).copy_to_mat(),
            Tol<double>::roundoff()
        );

    }

    template <template <typename> typename M>
    void Solve() {

        constexpr int n(20);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, n));
        TypedLinearSystem<M, double> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = n;
        args.target_rel_res = Tol<double>::krylov_conv_tol();

        GMRESSolve<M, double> gmres_solve(lin_sys, Tol<double>::roundoff(), args);

        gmres_solve.solve();
        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), Tol<double>::krylov_conv_tol());

    }

    template <template <typename> typename M>
    void Reset() {

        constexpr int n(20);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, n));
        TypedLinearSystem<M, double> lin_sys(A, b);

        SolveArgPkg args;
        args.max_iter = n;
        args.target_rel_res = Tol<double>::krylov_conv_tol();

        GMRESSolveTestingMock<M, double> test_mock(lin_sys, Tol<double>::roundoff(), args);

        test_mock.solve();
        if (*show_plots) { test_mock.view_relres_plot("log"); }
        
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_GT(test_mock.get_iteration(), 0);
        EXPECT_LE(test_mock.get_relres(), Tol<double>::krylov_conv_tol());

        test_mock.reset();
        ASSERT_FALSE(test_mock.check_converged());
        ASSERT_EQ(test_mock.get_iteration(), 0);
        if (*show_plots) { test_mock.view_relres_plot("log"); }

        // Check that all matrices are zero again and that krylov dim is back to 0
        EXPECT_EQ(test_mock.curr_kry_dim, 0);

        ASSERT_MATRIX_ZERO(test_mock.Q_kry_basis, Tol<double>::roundoff());
        ASSERT_VECTOR_ZERO(test_mock.H_k, Tol<double>::roundoff());
        ASSERT_MATRIX_IDENTITY(test_mock.H_Q, Tol<double>::roundoff());
        ASSERT_MATRIX_ZERO(test_mock.H_R, Tol<double>::roundoff());

        // Test 2nd solve
        test_mock.solve();
        if (*show_plots) { test_mock.view_relres_plot("log"); }
        
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_GT(test_mock.get_iteration(), 0);
        EXPECT_LE(test_mock.get_relres(), Tol<double>::krylov_conv_tol());

    }

    template <template <typename> typename M>
    void CheckCorrectDefaultMaxIter() {
    
        constexpr int n(7);
        M<double> A_n(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b_n(Vector<double>::Random(TestBase::bundle, n));
        TypedLinearSystem<M, double> lin_sys_n(A_n, b_n);
        GMRESSolveTestingMock<M, double> test_mock_n(lin_sys_n, Tol<double>::roundoff(), default_args);
        ASSERT_EQ(test_mock_n.max_iter, n);

        constexpr int m(53);
        M<double> A_m(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, m));
        Vector<double> b_m(Vector<double>::Random(TestBase::bundle, m));
        TypedLinearSystem<M, double> lin_sys_m(A_m, b_m);
        GMRESSolveTestingMock<M, double> test_mock_m(lin_sys_m, Tol<double>::roundoff(), default_args);
        ASSERT_EQ(test_mock_m.max_iter, m);

        constexpr int o(64);
        constexpr int non_default_iter(10);
        M<double> A_o(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, o, o));
        Vector<double> b_o(Vector<double>::Random(TestBase::bundle, o));
        TypedLinearSystem<M, double> lin_sys_o(A_o, b_o);
        SolveArgPkg non_default_args;
        non_default_args.max_iter = non_default_iter;
        GMRESSolveTestingMock<M, double> test_mock_o(lin_sys_o, Tol<double>::roundoff(), non_default_args);
        ASSERT_EQ(test_mock_o.max_iter, non_default_iter);

    }

    template <template <typename> typename M>
    void CheckErrorExceedDimension() {
    
        constexpr int n(7);
        M<double> A_n(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, n, n));
        Vector<double> b_n(Vector<double>::Random(TestBase::bundle, n));
        SolveArgPkg args;
        args.max_iter = 100;

        TypedLinearSystem<M, double> lin_sys_n(A_n, b_n);
        auto try_to_exceed_dim = [=]() {
            GMRESSolveTestingMock<M, double> test_mock_n(lin_sys_n, Tol<double>::roundoff(), args);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_to_exceed_dim);

    }


};

TEST_F(GMRESSolve_Component_Test, CheckConstruction5x5_SOLVER) {
    CheckConstruction<MatrixDense>(5);
    CheckConstruction<NoFillMatrixSparse>(5);
}

TEST_F(GMRESSolve_Component_Test, CheckConstruction64x64_SOLVER) {
    CheckConstruction<MatrixDense>(64);
    CheckConstruction<NoFillMatrixSparse>(64);
}

TEST_F(GMRESSolve_Component_Test, CheckCorrectDefaultMaxIter_SOLVER) {
    CheckCorrectDefaultMaxIter<MatrixDense>();
    CheckCorrectDefaultMaxIter<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, CheckErrorExceedDimension_SOLVER) {
    CheckErrorExceedDimension<MatrixDense>();
    CheckErrorExceedDimension<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, KrylovInitAndUpdate_SOLVER) {
    KrylovInitAndUpdate<MatrixDense>();
    KrylovInitAndUpdate<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, H_QR_Update_SOLVER) {
    H_QR_Update<MatrixDense>();
    H_QR_Update<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, Update_x_Back_Substitution_SOLVER) {
    Update_x_Back_Substitution<MatrixDense>();
    Update_x_Back_Substitution<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, KrylovLuckyBreakFirstIter_SOLVER) {
    KrylovLuckyBreakFirstIter<MatrixDense>();
    KrylovLuckyBreakFirstIter<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, KrylovLuckyBreakLaterIter_SOLVER) {
    KrylovLuckyBreakLaterIter<MatrixDense>();
    KrylovLuckyBreakLaterIter<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, KrylovLuckyBreakThroughSolve_SOLVER) {
    KrylovLuckyBreakThroughSolve<MatrixDense>();
    KrylovLuckyBreakThroughSolve<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, Solve_SOLVER) {
    Solve<MatrixDense>();
    Solve<NoFillMatrixSparse>();
}

TEST_F(GMRESSolve_Component_Test, Reset_SOLVER) {
    Reset<MatrixDense>();
    Reset<NoFillMatrixSparse>();
}
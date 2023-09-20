#include "../../../test.h"

#include "test_GMRES.h"

class GMRESComponentTest: public TestBase
{
public:

    template <template <typename> typename M>
    void CheckConstruction(
        const string &A_file_path,
        const string &b_file_path,
        const int &n
    ) {

        M<double> A = read_matrixCSV<M, double>(A_file_path);
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(b_file_path);
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, default_args);

        ASSERT_EQ(test_mock.max_kry_space_dim, n);
        ASSERT_EQ(test_mock.rho, (b - A*MatrixVector<double>::Ones(n)).norm());
        
        ASSERT_EQ(test_mock.Q_kry_basis.rows(), n);
        ASSERT_EQ(test_mock.Q_kry_basis.cols(), n);
        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.coeff(i, j), 0);
            }
        }

        ASSERT_EQ(test_mock.H.rows(), n+1);
        ASSERT_EQ(test_mock.H.cols(), n);
        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.H.coeff(i, j), 0);
            }
        }

        ASSERT_EQ(test_mock.Q_H.rows(), n+1);
        ASSERT_EQ(test_mock.Q_H.cols(), n+1);
        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n+1; ++j) {
                if (i == j) {
                    ASSERT_EQ(test_mock.Q_H.coeff(i, j), 1);
                } else {
                    ASSERT_EQ(test_mock.Q_H.coeff(i, j), 0);
                }
            }
        }

        ASSERT_EQ(test_mock.R_H.rows(), n+1);
        ASSERT_EQ(test_mock.R_H.cols(), n);
        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.R_H.coeff(i, j), 0);
            }
        }

    }

    template <template <typename> typename M>
    void KrylovInitAndUpdate() {

        const int n(5);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_5_toy.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "b_5_toy.csv");

        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, default_args);

        test_mock.typed_soln = MatrixVector<double>::Ones(n); // Manually instantiate initial guess
        MatrixVector<double> r_0 = b - A*MatrixVector<double>::Ones(n);

        // Create matrix to store previous basis vectors to ensure no change across iterations
        MatrixDense<double> Q_save = MatrixDense<double>::Zero(n, n);
        MatrixDense<double> H_save = MatrixDense<double>::Zero(n+1, n);

        // First update check first vector for basis is residual norm
        // and that Hessenberg first vector contructs next vector with entries
        test_mock.iterate_no_soln_solve();

        ASSERT_EQ(test_mock.Q_kry_basis.col(0), r_0/r_0.norm());
        MatrixVector<double> next_q = A*test_mock.Q_kry_basis.col(0);
        next_q -= test_mock.H.coeff(0, 0)*test_mock.Q_kry_basis.col(0);
        ASSERT_EQ(next_q.norm(), test_mock.H.coeff(1, 0));
        ASSERT_EQ(test_mock.next_q, next_q);

        // Save basis and H entries to check that they remain unchanged
        Q_save.col(0) = test_mock.Q_kry_basis.col(0);
        H_save.col(0) = test_mock.H.col(0);
        
        // Subsequent updates
        for (int k=1; k<n; ++k) {

            // Iterate Krylov subspace and Hessenberg
            test_mock.iterate_no_soln_solve();
            Q_save.col(k) = test_mock.Q_kry_basis.col(k);
            H_save.col(k) = test_mock.H.col(k);

            // Get newly generated basis vector
            MatrixVector<double> q = test_mock.Q_kry_basis.col(k);

            // Confirm that previous vectors are unchanged and are orthogonal to new one
            for (int j=0; j<k; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.col(j), Q_save.col(j));
                ASSERT_NEAR(test_mock.Q_kry_basis.col(j).dot(q), 0., dbl_error_acc);
            }

            // Confirm that Hessenberg matrix column corresponding to new basis vector
            // approximately constructs the next basis vector
            MatrixVector<double> h = test_mock.H.col(k);
            MatrixVector<double> construct_q = A*test_mock.Q_kry_basis.col(k);
            for (int i=0; i<=k; ++i) {
                ASSERT_EQ(test_mock.Q_kry_basis.col(i).dot(construct_q), h(i));
                construct_q -= h(i)*test_mock.Q_kry_basis.col(i);
            }
            EXPECT_EQ(construct_q.norm(), h(k+1));
            
            // Confirm that previous Hessenberg columns are untouched
            for (int j=0; j<=k; ++j) { ASSERT_EQ(test_mock.H.col(j), H_save.col(j)); }

        }
    
    }

    template <template <typename> typename M>
    void H_QR_Update() {

        const int n(5);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_5_toy.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "b_5_toy.csv");

        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, default_args);

        test_mock.typed_soln = MatrixVector<double>::Ones(n); // Manually instantiate initial guess
        MatrixVector<double> r_0 = b - A*MatrixVector<double>::Ones(n);

        // Fully create Hessenberg matrix
        test_mock.iterate_no_soln_solve();
        test_mock.iterate_no_soln_solve();
        test_mock.iterate_no_soln_solve();
        test_mock.iterate_no_soln_solve();
        test_mock.iterate_no_soln_solve();

        MatrixDense<double> save_Q_H = MatrixDense<double>::Zero(n+1, n+1);
        MatrixDense<double> save_R_H = MatrixDense<double>::Zero(n+1, n);

        for (int kry_dim=1; kry_dim<=n; ++kry_dim) {

            int k = kry_dim-1;

            // Set krylov dimension to kry_dim and update QR
            test_mock.kry_space_dim = kry_dim;
            test_mock.update_QR_fact();

            // Check that previous columns are unchanged by new update
            for (int i=0; i<k; ++i) {
                ASSERT_EQ(test_mock.Q_H.col(i), save_Q_H.col(i));
                ASSERT_EQ(test_mock.R_H.col(i), save_R_H.col(i));
            }

            // Save second last new basis vector and new column of R
            save_Q_H.col(k) = test_mock.Q_H.col(k);
            save_R_H.col(k) = test_mock.R_H.col(k);

            // Test that k+1 by k+1 block of Q_H is orthogonal
            MatrixDense<double> orthog_check = (test_mock.Q_H.block(0, 0, k+2, k+2)*
                                                test_mock.Q_H.block(0, 0, k+2, k+2).transpose());
            for (int i=0; i<k+1; ++i) {
                for (int j=0; j<k+1; ++j) {
                    if (i == j) {
                        ASSERT_NEAR(orthog_check.coeff(i, j), 1., u_dbl);
                    } else {
                        ASSERT_NEAR(orthog_check.coeff(i, j), 0., u_dbl);
                    }
                }
            }

            // Test that k+1 by k block of R_H is uppertriangular
            for (int i=0; i<k+1; ++i) {
                for (int j=0; j<i; ++j) {
                    ASSERT_EQ(test_mock.R_H.coeff(i, j), 0.);
                }
            }

            // Test that k+1 by k+1 block of Q_H is and k+1 by k block of R_H
            // constructs k+1 by k block of H
            MatrixDense<double> construct_H = (test_mock.Q_H.block(0, 0, k+2, k+2)*
                                               test_mock.R_H.block(0, 0, k+2, k+1));
            for (int i=0; i<k+1; ++i) {
                for (int j=0; j<k; ++j) {
                    ASSERT_NEAR(construct_H.coeff(i, j), test_mock.H.coeff(i, j), u_dbl);
                }
            }

        }

    }

    template <template <typename> typename M>
    void Update_x_Back_Substitution() {

        const int n(7);
        MatrixDense<double> Q = read_matrixCSV<MatrixDense, double>(solve_matrix_dir + "Q_8_backsub.csv");
        MatrixDense<double> R = read_matrixCSV<MatrixDense, double>(solve_matrix_dir + "R_8_backsub.csv");
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_7_dummy_backsub.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "b_7_dummy_backsub.csv");

        // Set initial guess to zeros such that residual is just b
        MatrixVector<double> x_0 = MatrixVector<double>::Zero(n);
        SolveArgPkg args;
        args.init_guess = x_0;
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, args);

        // Set test_mock krylov basis to the identity to have typed_soln be directly the solved coefficients
        // of the back substitution
        test_mock.Q_kry_basis = MatrixDense<double>::Identity(n, n);

        // Set premade Q R decomposition for H
        test_mock.Q_H = Q;
        test_mock.R_H = R;

        // Test that for each possible Hessenberg size determined by the Krylov subspace dimension
        // that the coefficient solution matches the pre-determined correct one from MATLAB solving
        for (int kry_dim=1; kry_dim<=n; ++kry_dim) {

            // Set Krylov subspace dim
            test_mock.kry_space_dim = kry_dim;
            
            // Load test solution
            MatrixVector<double> test_soln = read_matrixCSV<MatrixVector, double>(
                solve_matrix_dir + "x_" + std::to_string(kry_dim) + "_backsub.csv"
            );

            // )Solve with backsubstitution
            test_mock.update_x_minimizing_res();

            // Check if coefficient solution matches to within tolerable error
            for (int i=0; i<kry_dim; ++i) {
                ASSERT_NEAR(test_mock.typed_soln(i), test_soln(i), dbl_error_acc);
            }

        }

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakFirstIter () {

        const int n(5);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_5_easysoln.csv");
        MatrixVector<double> b = read_matrixCSV<M, double>(solve_matrix_dir + "b_5_easysoln.csv");

        MatrixVector<double> soln = MatrixVector<double>::Ones(n); // Instantiate initial guess as true solution
        SolveArgPkg args;
        args.target_rel_res = conv_tol_dbl;
        args.init_guess = soln;
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, args);

        // Attempt to update subspace and Hessenberg
        test_mock.iterate();

        // Check basis Q and H are empty and Krylov hasn't been updated since already
        // hit the lucky break so can't build subspace and check that terminated but
        // not converged
        EXPECT_FALSE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.kry_space_dim, 0);
        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.coeff(i, j), 0);
            }
        }
        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.H.coeff(i, j), 0);
            }
        }

        // Attempt to solve and check that iteration does not occur since
        // should be terminated already but that convergence is updated
        test_mock.solve();
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_EQ(test_mock.get_iteration(), 0);

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakLaterIter() {

        constexpr int n(5);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_5_easysoln.csv");
        MatrixVector<double> b = read_matrixCSV<M, double>(solve_matrix_dir + "b_5_easysoln.csv");

        MatrixVector<double> soln = MatrixVector<double>::Zero(n); // Initialize as near solution
        soln(0) = 1;
        SolveArgPkg args;
        args.init_guess = soln;
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, args);

        // Attempt to update subspace and convergence twice
        test_mock.iterate();
        test_mock.iterate();
        
        // Check basis Q has one normalized basis vector and others are empty and
        // has been marked terminated but not converged since we want to delay that
        // check for LinearSolve
        EXPECT_FALSE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.kry_space_dim, 1);
        EXPECT_NEAR(test_mock.Q_kry_basis.col(0).norm(), 1, gamma(n, u_dbl));
        for (int i=0; i<n; ++i) {
            for (int j=1; j<n; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.coeff(i, j), 0);
            }
        }

    }

    template <template <typename> typename M>
    void KrylovLuckyBreakThroughSolve() {

        constexpr int n(5);
        M<double> A = read_matrixCSV<M, double>(solve_matrix_dir + "A_5_easysoln.csv");
        MatrixVector<double> b = read_matrixCSV<MatrixVector, double>(solve_matrix_dir + "b_5_easysoln.csv");

        MatrixVector<double> soln = MatrixVector<double>::Zero(n); // Initialize as near solution
        soln(0) = 1;
        SolveArgPkg args;
        args.init_guess = soln;
        args.target_rel_res = conv_tol_dbl;
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, args);

        // Attempt to update and solve through solve of LinearSolve
        test_mock.solve();
        
        // Check we have terminated at second iteration and have converged
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.get_iteration(), 1);

        // Check that subspace has not gone beyond 1 dimension and that krylov basis
        // as expected to have only a single column
        EXPECT_EQ(test_mock.kry_space_dim, 1);
        EXPECT_NEAR(test_mock.Q_kry_basis.col(0).norm(), 1, gamma(n, u_dbl));
        for (int i=0; i<n; ++i) {
            for (int j=1; j<n; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.coeff(i, j), 0);
            }
        }

    }

    template <template <typename> typename M>
    void Solve() {

        constexpr int n(20);
        M<double> A = M<double>::Random(n, n);
        MatrixVector<double> b = MatrixVector<double>::Random(n);

        SolveArgPkg args;
        args.max_iter = n;
        args.target_rel_res = conv_tol_dbl;
        GMRESSolve<M, double> gmres_solve(A, b, u_dbl, args);

        gmres_solve.solve();
        if (*show_plots) { gmres_solve.view_relres_plot("log"); }
        
        EXPECT_TRUE(gmres_solve.check_converged());
        EXPECT_LE(gmres_solve.get_relres(), conv_tol_dbl);

    }

    template <template <typename> typename M>
    void Reset() {

        constexpr int n(20);
        M<double> A = M<double>::Random(n, n);
        MatrixVector<double> b = MatrixVector<double>::Random(n);

        SolveArgPkg args;
        args.max_iter = n;
        args.target_rel_res = conv_tol_dbl;
        GMRESSolveTestingMock<M, double> test_mock(A, b, u_dbl, args);

        test_mock.solve();
        if (*show_plots) { test_mock.view_relres_plot("log"); }
        
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_GT(test_mock.get_iteration(), 0);
        EXPECT_LE(test_mock.get_relres(), conv_tol_dbl);

        test_mock.reset();
        ASSERT_FALSE(test_mock.check_converged());
        ASSERT_EQ(test_mock.get_iteration(), 0);
        if (*show_plots) { test_mock.view_relres_plot("log"); }

        // Check that all matrices are zero again and that krylov dim is back to 0
        EXPECT_EQ(test_mock.kry_space_dim, 0);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.Q_kry_basis.coeff(i, j), 0.);
            }
        }

        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.H.coeff(i, j), 0.);
            }
        }

        for (int i=0; i<n+1; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == j) {
                    ASSERT_EQ(test_mock.Q_H.coeff(i, j), 1.);
                } else {
                    ASSERT_EQ(test_mock.Q_H.coeff(i, j), 0.);
                }
            }
        }

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mock.R_H.coeff(i, j), 0.);
            }
        }

        // Test 2nd solve
        test_mock.solve();
        if (*show_plots) { test_mock.view_relres_plot("log"); }
        
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_GT(test_mock.get_iteration(), 0);
        EXPECT_LE(test_mock.get_relres(), conv_tol_dbl);

    }

};

TEST_F(GMRESComponentTest, CheckConstruction5x5_Dense) {
    CheckConstruction<MatrixDense>(solve_matrix_dir+"A_5_toy.csv", solve_matrix_dir+"b_5_toy.csv", 5);
}
TEST_F(GMRESComponentTest, CheckConstruction5x5_Sparse) {
    CheckConstruction<MatrixSparse>(solve_matrix_dir+"A_5_toy.csv", solve_matrix_dir+"b_5_toy.csv", 5);
}

TEST_F(GMRESComponentTest, CheckConstruction64x64_Dense) {
    CheckConstruction<MatrixDense>(solve_matrix_dir+"A_64_toy.csv", solve_matrix_dir+"b_64_toy.csv", 64);
}
TEST_F(GMRESComponentTest, CheckConstruction64x64_Sparse) {
    CheckConstruction<MatrixSparse>(solve_matrix_dir+"A_64_toy.csv", solve_matrix_dir+"b_64_toy.csv", 64);
}

TEST_F(GMRESComponentTest, CheckCorrectDefaultMaxIter_Both) {
    
    constexpr int n(7);
    MatrixDense<double> A_n = MatrixDense<double>::Random(n, n);
    MatrixVector<double> b_n = MatrixVector<double>::Random(n);
    GMRESSolveTestingMock<MatrixDense, double> test_mock_n_dense(
        A_n, b_n, u_dbl, default_args
    );
    ASSERT_EQ(test_mock_n_dense.max_iter, n);
    GMRESSolveTestingMock<MatrixSparse, double> test_mock_n_sparse(
        A_n.sparseView(), b_n, u_dbl, default_args
    );
    ASSERT_EQ(test_mock_n_sparse.max_iter, n);

    constexpr int m(53);
    MatrixDense<double> A_m = MatrixDense<double>::Random(m, m);
    MatrixVector<double> b_m = MatrixVector<double>::Random(m);
    GMRESSolveTestingMock<MatrixDense, double> test_mock_m_dense(
        A_m, b_m, u_dbl, default_args
    );
    ASSERT_EQ(test_mock_m_dense.max_iter, m);
    GMRESSolveTestingMock<MatrixSparse, double> test_mock_m_sparse(
        A_m.sparseView(), b_m, u_dbl, default_args
    );
    ASSERT_EQ(test_mock_m_sparse.max_iter, m);

    constexpr int o(64);
    constexpr int non_default_iter(10);
    MatrixDense<double> A_o = MatrixDense<double>::Random(o, o);
    MatrixVector<double> b_o = MatrixVector<double>::Random(o);
    SolveArgPkg non_default_args;
    non_default_args.max_iter = non_default_iter;
    GMRESSolveTestingMock<MatrixDense, double> test_mock_o_dense(
        A_o, b_o, u_dbl, non_default_args
    );
    ASSERT_EQ(test_mock_o_dense.max_iter, non_default_iter);
    GMRESSolveTestingMock<MatrixSparse, double> test_mock_o_sparse(
        A_o.sparseView(), b_o, u_dbl, non_default_args
    );
    ASSERT_EQ(test_mock_o_sparse.max_iter, non_default_iter);

}

TEST_F(GMRESComponentTest, CheckErrorExceedDimension_Both) {
    
    constexpr int n(7);
    MatrixDense<double> A_n = MatrixDense<double>::Random(n, n);
    MatrixVector<double> b_n = MatrixVector<double>::Random(n);
    SolveArgPkg args;
    args.max_iter = 100;

    try {
        GMRESSolveTestingMock<MatrixDense, double> test_mock_n(A_n, b_n, u_dbl, args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

    try {
        GMRESSolveTestingMock<MatrixSparse, double> test_mock_n(A_n.sparseView(), b_n, u_dbl, args);
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }

}

TEST_F(GMRESComponentTest, KrylovInitAndUpdate_Dense) { KrylovInitAndUpdate<MatrixDense>(); }
TEST_F(GMRESComponentTest, KrylovInitAndUpdate_Sparse) { KrylovInitAndUpdate<MatrixSparse>(); }

TEST_F(GMRESComponentTest, H_QR_Update_Dense) { H_QR_Update<MatrixDense>(); }
TEST_F(GMRESComponentTest, H_QR_Update_Sparse) { H_QR_Update<MatrixSparse>(); }

TEST_F(GMRESComponentTest, Update_x_Back_Substitution_Dense) { Update_x_Back_Substitution<MatrixDense>(); }
TEST_F(GMRESComponentTest, Update_x_Back_Substitution_Sparse) { Update_x_Back_Substitution<MatrixSparse>(); }

TEST_F(GMRESComponentTest, KrylovLuckyBreakFirstIter_Dense) { KrylovLuckyBreakFirstIter<MatrixDense>(); }
TEST_F(GMRESComponentTest, KrylovLuckyBreakFirstIter_Sparse) { KrylovLuckyBreakFirstIter<MatrixSparse>(); }

TEST_F(GMRESComponentTest, KrylovLuckyBreakLaterIter_Dense) { KrylovLuckyBreakLaterIter<MatrixDense>(); }
TEST_F(GMRESComponentTest, KrylovLuckyBreakLaterIter_Sparse) { KrylovLuckyBreakLaterIter<MatrixSparse>(); }

TEST_F(GMRESComponentTest, KrylovLuckyBreakThroughSolve_Dense) { KrylovLuckyBreakThroughSolve<MatrixDense>(); }
TEST_F(GMRESComponentTest, KrylovLuckyBreakThroughSolve_Sparse) { KrylovLuckyBreakThroughSolve<MatrixSparse>(); }

TEST_F(GMRESComponentTest, Solve_Dense) { Solve<MatrixDense>(); }
TEST_F(GMRESComponentTest, Solve_Sparse) { Solve<MatrixSparse>(); }

TEST_F(GMRESComponentTest, Reset_Dense) { Reset<MatrixDense>(); }
TEST_F(GMRESComponentTest, Reset_Sparse) { Reset<MatrixSparse>(); }
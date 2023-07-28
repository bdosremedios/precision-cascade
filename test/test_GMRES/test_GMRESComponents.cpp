#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;

class GMRESComponentTest: public testing::Test {

    protected:
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 4*pow(2, -52); // Set as 4 times machines epsilon

};

TEST_F(GMRESComponentTest, CheckConstruction5x5) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_toy.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_toy.csv");
    GMRESSolveTestingMock<double, double> test_mock(A, b, double_tolerance);
    ASSERT_EQ(test_mock.max_krylov_subspace_dim, 5);
    ASSERT_EQ(test_mock.rho, (b - A*MatrixXd::Ones(5, 1)).norm());
    
    ASSERT_EQ(test_mock.Q_kry_basis.rows(), 5);
    ASSERT_EQ(test_mock.Q_kry_basis.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis(i, j), 0);
        }
    }

    ASSERT_EQ(test_mock.H.rows(), 6);
    ASSERT_EQ(test_mock.H.cols(), 5);
    for (int i=0; i<6; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.H(i, j), 0);
        }
    }

    ASSERT_EQ(test_mock.Q_H.rows(), 6);
    ASSERT_EQ(test_mock.Q_H.cols(), 6);
    for (int i=0; i<6; ++i) {
        for (int j=0; j<6; ++j) {
            if (i == j) {
                EXPECT_EQ(test_mock.Q_H(i, j), 1);
            } else {
                EXPECT_EQ(test_mock.Q_H(i, j), 0);
            }
        }
    }

    ASSERT_EQ(test_mock.R_H.rows(), 6);
    ASSERT_EQ(test_mock.R_H.cols(), 5);
    for (int i=0; i<6; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.R_H(i, j), 0);
        }
    }

}

TEST_F(GMRESComponentTest, CheckConstruction64x64) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "conv_diff_64_b.csv");
    GMRESSolveTestingMock<double, double> test_mock(A, b, double_tolerance);
    ASSERT_EQ(test_mock.max_krylov_subspace_dim, 64);
    ASSERT_EQ(test_mock.rho, (b - A*MatrixXd::Ones(64, 1)).norm());
    
    ASSERT_EQ(test_mock.Q_kry_basis.rows(), 64);
    ASSERT_EQ(test_mock.Q_kry_basis.cols(), 64);
    for (int i=0; i<64; ++i) {
        for (int j=0; j<64; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis(i, j), 0);
        }
    }

    ASSERT_EQ(test_mock.H.rows(), 65);
    ASSERT_EQ(test_mock.H.cols(), 64);
    for (int i=0; i<65; ++i) {
        for (int j=0; j<64; ++j) {
            EXPECT_EQ(test_mock.H(i, j), 0);
        }
    }

    ASSERT_EQ(test_mock.Q_H.rows(), 65);
    ASSERT_EQ(test_mock.Q_H.cols(), 65);
    for (int i=0; i<65; ++i) {
        for (int j=0; j<65; ++j) {
            if (i == j) {
                EXPECT_EQ(test_mock.Q_H(i, j), 1);
            } else {
                EXPECT_EQ(test_mock.Q_H(i, j), 0);
            }
        }
    }

    ASSERT_EQ(test_mock.R_H.rows(), 65);
    ASSERT_EQ(test_mock.R_H.cols(), 64);
    for (int i=0; i<65; ++i) {
        for (int j=0; j<64; ++j) {
            EXPECT_EQ(test_mock.R_H(i, j), 0);
        }
    }

}

TEST_F(GMRESComponentTest, CheckAssertSquareness) {
    
    Matrix<double, Dynamic, Dynamic> A = MatrixXd(6, 5);
    Matrix<double, Dynamic, Dynamic> b = MatrixXd(6, 1);
    try {
        GMRESSolveTestingMock<double, double> test_mock(A, b, double_tolerance);
        FAIL();
    } catch (runtime_error e) {
        ;
    }
 
}

TEST_F(GMRESComponentTest, KrylovInstantiationAndUpdates) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_toy.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_toy.csv");
    GMRESSolveTestingMock<double, double> test_mock(A, b, double_tolerance);
    test_mock.x = MatrixXd::Ones(5, 1); // Manually instantiate initial guess
    Matrix<double, Dynamic, 1> r_0 = b - A*MatrixXd::Ones(5, 1);

    // Create matrix to store previous basis vectors to ensure no change across iterations
    MatrixXd Q_save = MatrixXd::Zero(5, 5);
    MatrixXd H_save = MatrixXd::Zero(6, 5);

    // First update check first vector for basis is residual norm
    // and that Hessenberg first vector contructs next vector with entries
    test_mock.iterate_no_soln_solve();

    ASSERT_EQ(test_mock.Q_kry_basis.col(0), r_0/r_0.norm());
    Matrix<double, Dynamic, 1> next_q = A*test_mock.Q_kry_basis.col(0);
    next_q -= test_mock.H(0, 0)*test_mock.Q_kry_basis.col(0);
    ASSERT_EQ(next_q.norm(), test_mock.H(1, 0));
    ASSERT_EQ(test_mock.next_q, next_q);

    // Save basis and H entries to check that they remain unchanged
    Q_save(all, 0) = test_mock.Q_kry_basis.col(0);
    H_save(all, 0) = test_mock.H.col(0);
    
    // Subsequent updates
    for (int k=1; k<5; ++k) {

        // Iterate Krylov subspace and Hessenberg
        test_mock.iterate_no_soln_solve();
        Q_save(all, k) = test_mock.Q_kry_basis.col(k);
        H_save(all, k) = test_mock.H.col(k);

        // Get newly generated basis vector
        Matrix<double, Dynamic, 1> q = test_mock.Q_kry_basis.col(k);

        // Confirm that previous vectors are unchanged and are orthogonal to new one
        for (int j=0; j<k; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis.col(j), Q_save(all, j));
            EXPECT_NEAR(test_mock.Q_kry_basis.col(j).dot(q), 0, 1e-13);
        }

        // Confirm that Hessenberg matrix column corresponding to new basis vector
        // approximately constructs the next basis vector
        Matrix<double, Dynamic, 1> h = test_mock.H.col(k);
        Matrix<double, Dynamic, 1> construct_q = A*test_mock.Q_kry_basis.col(k);
        for (int i=0; i<=k; ++i) {
            EXPECT_EQ(test_mock.Q_kry_basis.col(i).dot(construct_q), h(i));
            construct_q -= h(i)*test_mock.Q_kry_basis.col(i);
        }
        EXPECT_EQ(construct_q.norm(), h(k+1));
        
        // Confirm that previous Hessenberg columns are untouched
        for (int j=0; j<=k; ++j) {
            EXPECT_EQ(test_mock.H.col(j), H_save(all, j));
        }

    }
    
}

TEST_F(GMRESComponentTest, H_QR_Update) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_toy.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_toy.csv");
    GMRESSolveTestingMock<double, double> test_mock(A, b, double_tolerance);
    test_mock.x = MatrixXd::Ones(5, 1); // Manually instantiate initial guess
    Matrix<double, Dynamic, 1> r_0 = b - A*MatrixXd::Ones(5, 1);

    // Fully create Hessenberg matrix
    test_mock.iterate_no_soln_solve();
    test_mock.iterate_no_soln_solve();
    test_mock.iterate_no_soln_solve();
    test_mock.iterate_no_soln_solve();
    test_mock.iterate_no_soln_solve();

    Matrix<double, 6, 4> save_Q_H;
    Matrix<double, 6, 4> save_R_H;

    for (int kry_dim=1; kry_dim<=4; ++kry_dim) {

        int k = kry_dim-1;

        // Set krylov dimension to kry_dim and update QR
        test_mock.krylov_subspace_dim = kry_dim;
        test_mock.update_QR_fact();

        // Check that previous columns are unchanged by new update
        for (int i=0; i<k; ++i) {
            EXPECT_EQ(test_mock.Q_H.col(i), save_Q_H.col(i));
            EXPECT_EQ(test_mock.R_H.col(i), save_R_H.col(i));
        }

        // Save second last new basis vector and new column of R
        save_Q_H.col(k) = test_mock.Q_H.col(k);
        save_R_H.col(k) = test_mock.R_H.col(k);

        // Test that k+1 by k+1 block of Q_H is orthogonal
        Matrix<double, Dynamic, Dynamic> orthog_check = test_mock.Q_H.block(0, 0, k+2, k+2)*
                                                        test_mock.Q_H.block(0, 0, k+2, k+2).transpose();
        for (int i=0; i<k+1; ++i) {
            for (int j=0; j<k+1; ++j) {
                if (i == j) {
                    EXPECT_NEAR(orthog_check(i, j), 1, double_tolerance);
                } else {
                    EXPECT_NEAR(orthog_check(i, j), 0, double_tolerance);
                }
            }
        }
        
        // Test that k+1 by k block of R_H is uppertriangular
        for (int j=0; j<k; ++j) {
            for (int i=k+1; i>j; --i) {
                EXPECT_EQ(test_mock.R_H(i, j), 0);
            }
        }

        // Test that k+1 by k+1 block of Q_H is and k+1 by k block of R_H
        // constructs k+1 by k block of H
        Matrix<double, Dynamic, Dynamic> construct_H = test_mock.Q_H.block(0, 0, k+2, k+2)*
                                                       test_mock.R_H.block(0, 0, k+2, k+1);
        for (int i=0; i<k+1; ++i) {
            for (int j=0; j<k; ++j) {
                EXPECT_NEAR(construct_H(i, j), test_mock.H(i, j), double_tolerance);
            }
        }

    }

}

TEST_F(GMRESComponentTest, Update_x_Back_Substitution) {
    
    Matrix<double, Dynamic, Dynamic> Q = read_matrix_csv<double>(matrix_dir + "Q_8_backsub.csv");
    Matrix<double, Dynamic, Dynamic> R = read_matrix_csv<double>(matrix_dir + "R_8_backsub.csv");
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_7_dummy_backsub.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_7_dummy_backsub.csv");

    // Set initial guess to zeros such that residual is just b
    Matrix<double, 7, 1> x_0 = MatrixXd::Zero(7, 1);
    GMRESSolveTestingMock<double, double> test_mock(A, b, x_0, double_tolerance);

    // Set test_mock krylov basis to the identity to have x be directly the solved coefficients
    // of the back substitution
    test_mock.Q_kry_basis = MatrixXd::Identity(7, 7);

    // Set premade Q R decomposition for H
    test_mock.Q_H = Q;
    test_mock.R_H = R;

    // Test that for each possible Hessenberg size determined by the Krylov subspace dimension
    // that the coefficient solution matches the pre-determined correct one from MATLAB solving
    for (int kry_dim=1; kry_dim<=7; ++kry_dim) {

        // Set Krylov subspace dim
        test_mock.krylov_subspace_dim = kry_dim;
        
        // Load test solution
        string solution_path = matrix_dir + "x_" + std::to_string(kry_dim) + "_backsub.csv";
        Matrix<double, Dynamic, 1> test_soln = read_matrix_csv<double>(solution_path);

        // Solve with backsubstitution
        test_mock.update_x_minimizing_res();

        // Check if coefficient solution matches to within double tolerance
        for (int i=0; i<kry_dim; ++i) {
            EXPECT_NEAR(test_mock.x(i), test_soln(i), 1e-13);
        }

    }

}

TEST_F(GMRESComponentTest, KrylovLuckyBreakFirstIter) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_easysoln.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_easysoln.csv");
    MatrixXd soln = MatrixXd::Ones(5, 1); // Instantiate initial guess as true solution
    GMRESSolveTestingMock<double, double> test_mock(A, b, soln, double_tolerance);

    // Attempt to update subspace and Hessenberg
    test_mock.iterate();
    
    // Check basis Q and H are empty and Krylov hasn't been updated since already
    // hit the lucky break so can't build subspace and check that terminated but
    // not converged
    EXPECT_FALSE(test_mock.check_converged());
    EXPECT_TRUE(test_mock.check_terminated());
    EXPECT_EQ(test_mock.krylov_subspace_dim, 0);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis(i, j), 0);
        }
    }
    for (int i=0; i<6; ++i) {
            for (int j=0; j<5; ++j) {
                EXPECT_EQ(test_mock.H(i, j), 0);
        }
    }

    // Attempt to solve and check that iteration does not occur since
    // should be terminated already but that convergence is updated
    test_mock.solve(100, 1e-10);
    EXPECT_TRUE(test_mock.check_converged());
    EXPECT_EQ(test_mock.get_iteration(), 0);

}

TEST_F(GMRESComponentTest, KrylovLuckyBreakLaterIter) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_easysoln.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_easysoln.csv");
    MatrixXd soln = MatrixXd::Zero(5, 1); // Initialize as near solution
    soln(0) = 1;
    GMRESSolveTestingMock<double, double> test_mock(A, b, soln, double_tolerance);

    // Attempt to update subspace and convergence twice
    test_mock.iterate();
    test_mock.iterate();
    
    // Check basis Q has one normalized basis vector and others are empty and
    // has been marked terminated but not converged since we want to delay that
    // check for LinearSolve
    EXPECT_FALSE(test_mock.check_converged());
    EXPECT_TRUE(test_mock.check_terminated());
    EXPECT_EQ(test_mock.krylov_subspace_dim, 1);
    EXPECT_NEAR(test_mock.Q_kry_basis.col(0).norm(), 1, double_tolerance);
    for (int i=0; i<5; ++i) {
        for (int j=1; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis(i, j), 0);
        }
    }
    
}

TEST_F(GMRESComponentTest, KrylovLuckyBreakThroughSolve) {
    
    Matrix<double, Dynamic, Dynamic> A = read_matrix_csv<double>(matrix_dir + "A_5_easysoln.csv");
    Matrix<double, Dynamic, Dynamic> b = read_matrix_csv<double>(matrix_dir + "b_5_easysoln.csv");
    MatrixXd soln = MatrixXd::Zero(5, 1); // Initialize as near solution
    soln(0) = 1;
    GMRESSolveTestingMock<double, double> test_mock(A, b, soln, double_tolerance);

    // Attempt to update and solve through solve of LinearSolve
    test_mock.solve(5, 1e-10);
    
    // Check we have terminated at second iteration and have converged
    EXPECT_TRUE(test_mock.check_converged());
    EXPECT_TRUE(test_mock.check_terminated());
    EXPECT_EQ(test_mock.get_iteration(), 1);

    // Check that subspace has not gone beyond 1 dimension and that krylov basis
    // as expected to have only a single column
    EXPECT_EQ(test_mock.krylov_subspace_dim, 1);
    EXPECT_NEAR(test_mock.Q_kry_basis.col(0).norm(), 1, double_tolerance);
    for (int i=0; i<5; ++i) {
        for (int j=1; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kry_basis(i, j), 0);
        }
    }
    
}
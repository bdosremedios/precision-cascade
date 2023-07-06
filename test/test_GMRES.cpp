#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using mxread::MatrixReader;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;
using Eigen::placeholders::all;

class GMRESTest: public testing::Test {

    protected:
        MatrixReader mr;
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
        double double_tolerance = 5*2.22e-16;

    public:
        GMRESTest() {
            mr = MatrixReader();
        }
        ~GMRESTest() = default;

};

TEST_F(GMRESTest, CheckConstruction) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "A_5.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "b_5.csv");
    GMRESSolveTestingMock<double> test_mock(A, b, 8.88e-16);
    
    ASSERT_EQ(test_mock.Q_kbasis.rows(), 5);
    ASSERT_EQ(test_mock.Q_kbasis.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kbasis(i, j), 0);
        }
    }

    ASSERT_EQ(test_mock.H.rows(), 6);
    ASSERT_EQ(test_mock.H.cols(), 5);
    for (int i=0; i<6; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.H(i, j), 0);
        }
    }

}

TEST_F(GMRESTest, KrylovInstantiationAndUpdates) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "A_5.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "b_5.csv");
    GMRESSolveTestingMock<double> test_mock(A, b, double_tolerance);
    test_mock.x = MatrixXd::Ones(5, 1); // Manually instantiate initial guess
    Matrix<double, Dynamic, 1> r_0 = b - A*MatrixXd::Ones(5, 1);

    // Create matrix to store previous basis vectors to ensure no change across iterations
    MatrixXd Q_save = MatrixXd::Zero(5, 5);
    MatrixXd H_save = MatrixXd::Zero(6, 5);

    // First update
    test_mock.update_subspace_and_convergence();
    Q_save(all, 0) = test_mock.Q_kbasis.col(0);
    EXPECT_EQ(test_mock.Q_kbasis.col(0), r_0/r_0.norm());
    
    // Subsequent updates
    for (int i=1; i<5; ++i) {
        
        // Update subspace and convergence
        test_mock.update_subspace_and_convergence();
        Q_save(all, i) = test_mock.Q_kbasis.col(i);
        H_save(all, i-1) = test_mock.H.col(i-1);

        // Get newly generated basis vector
        Matrix<double, Dynamic, 1> q = test_mock.Q_kbasis.col(i);

        // Confirm that previous vectors are unchanged and are orthogonal (to
        // within 4 double machine epsilon) to new one
        for (int j=0; j<i; ++j) {
            EXPECT_EQ(test_mock.Q_kbasis.col(j), Q_save(all, j));
            EXPECT_NEAR(test_mock.Q_kbasis.col(j).dot(q), 0, double_tolerance);
        }

        // Confirm that Hessenberg matrix column corresponding to new basis vector
        // approximately constructs the basis vector
        Matrix<double, Dynamic, 1> h = test_mock.H.col(i-1);
        Matrix<double, Dynamic, 1> construct_q = A*test_mock.Q_kbasis.col(i-1);
        for (int k=0; k<i; ++k) {
            construct_q -= h(k)*test_mock.Q_kbasis.col(k);
        }
        construct_q /= h(i);
        EXPECT_EQ(construct_q, q);
        
        // Confirm that previous Hessenberg columns are untouched
        for (int j=0; j<i-1; ++j) {
            EXPECT_EQ(test_mock.H.col(j), H_save(all, j));
        }

    }
    
}

TEST_F(GMRESTest, KrylovLuckyBreakFirstIter) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "A_5_easysoln.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "b_5.csv");
    GMRESSolveTestingMock<double> test_mock(A, b, double_tolerance);
    MatrixXd soln = MatrixXd::Ones(5, 1);
    test_mock.x = soln; // Manually instantiate initial guess as true solution

    // Attempt to update subspace and convergence
    test_mock.update_subspace_and_convergence();
    
    // Check basis Q and H are empty and Krylov hasn't been updated since already
    // hit the lucky break so can't build subspace
    EXPECT_TRUE(test_mock.check_converged());
    EXPECT_EQ(test_mock.krylov_subspace_dim, 0);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kbasis(i, j), 0);
        }
    }
    for (int i=0; i<6; ++i) {
            for (int j=0; j<5; ++j) {
                EXPECT_EQ(test_mock.H(i, j), 0);
        }
    }
}

TEST_F(GMRESTest, KrylovLuckyBreakLaterIter) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "A_5_easysoln.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "b_5.csv");
    GMRESSolveTestingMock<double> test_mock(A, b, double_tolerance);
    MatrixXd soln = MatrixXd::Zero(5, 1);
    soln(0) = 1;
    test_mock.x = soln; // Manually instantiate initial guess as true solution

    // Attempt to update subspace and convergence
    test_mock.update_subspace_and_convergence();
    test_mock.update_subspace_and_convergence();
    
    // Check basis Q has one normalized basis vector and others are empty
    EXPECT_TRUE(test_mock.check_converged());
    EXPECT_EQ(test_mock.krylov_subspace_dim, 1);
    EXPECT_NEAR(test_mock.Q_kbasis.col(0).norm(), 1, double_tolerance);
    for (int i=0; i<5; ++i) {
        for (int j=1; j<5; ++j) {
            EXPECT_EQ(test_mock.Q_kbasis(i, j), 0);
        }
    }
    
}

TEST_F(GMRESTest, SolveConvDiff64) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "conv_diff_64_b.csv");
    GMRESSolve<double> gmres_solve_d(A, b, 8.88e-16);
    // gmres_solve_d.solve();
    // gmres_solve_d.view_relres_plot("log");

}

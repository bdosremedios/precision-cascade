#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <string>
#include <memory>
#include <iostream>

using Eigen::MatrixXd;

using read_matrix::read_matrix_csv;

using std::make_shared;
using std::string;
using std::cout, std::endl;

class PGMRESTest: public TestBase {};

TEST_F(PGMRESTest, TestLeftPreconditioning_RandA45) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, Dynamic, Dynamic> Ainv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl, std::make_shared<MatrixInverse<double>>(Ainv), 45, conv_tol_dbl
    );

    pgmres_solve.solve();
    pgmres_solve.view_relres_plot("log");
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 1);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

}

TEST_F(PGMRESTest, TestRightPreconditioning_RandA45) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_inv_45.csv"));
    Matrix<double, Dynamic, Dynamic> Ainv(read_matrix_csv<double>(solve_matrix_dir + "Ainv_inv_45.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_inv_45.csv"));GMRESSolve<double> pgmres_solve(
        A, b, u_dbl,
        std::make_shared<NoPreconditioner<double>>(), std::make_shared<MatrixInverse<double>>(Ainv),
        45, conv_tol_dbl
    );

    pgmres_solve.solve();
    pgmres_solve.view_relres_plot("log");
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 1);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

}

TEST_F(PGMRESTest, TestLeftPreconditioning_3eigs) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_saddle.csv"));
    Matrix<double, Dynamic, Dynamic> Ainv(read_matrix_csv<double>(solve_matrix_dir + "A_25_invprecond_saddle.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_saddle.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl, std::make_shared<MatrixInverse<double>>(Ainv), 25, conv_tol_dbl
    );

    pgmres_solve.solve();
    pgmres_solve.view_relres_plot("log");
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 3);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_25_saddle.csv"));
    EXPECT_LE((pgmres_solve.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}

TEST_F(PGMRESTest, TestRightPreconditioning_3eigs) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "A_25_saddle.csv"));
    Matrix<double, Dynamic, Dynamic> Ainv(read_matrix_csv<double>(solve_matrix_dir + "A_25_invprecond_saddle.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "b_25_saddle.csv"));
    GMRESSolve<double> pgmres_solve(
        A, b, u_dbl,
        std::make_shared<NoPreconditioner<double>>(), std::make_shared<MatrixInverse<double>>(Ainv),
        25, conv_tol_dbl
    );

    pgmres_solve.solve();
    pgmres_solve.view_relres_plot("log");
    
    EXPECT_EQ(pgmres_solve.get_iteration(), 3);
    EXPECT_TRUE(pgmres_solve.check_converged());
    EXPECT_LE(pgmres_solve.get_relres(), conv_tol_dbl);

    // Check that matches MATLAB gmres solution within the difference of twice conv_tol_dbl
    Matrix<double, Dynamic, Dynamic> x_test(read_matrix_csv<double>(solve_matrix_dir + "x_25_saddle.csv"));
    EXPECT_LE((pgmres_solve.get_soln() - x_test).norm()/(x_test.norm()), 2*conv_tol_dbl);

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/FP_GMRES_IR.h"

#include <string>

using read_matrix::read_matrix_csv;

using Eigen::half;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using Eigen::MatrixXf;
using Eigen::MatrixXd;

using std::string;
using std::cout, std::endl;

class FP_GMRES_IRTest: public TestBase {

    // public:
    //     int max_iter = 2000;
    //     int fail_iter = 400;

};

TEST_F(FP_GMRES_IRTest, BasicTest) {

    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    FP_GMRES_IR_Solve<double> fp_gmres_ir_solve_d(A, b, u_dbl, 10, 10, conv_tol_dbl);
    fp_gmres_ir_solve_d.solve();
    fp_gmres_ir_solve_d.view_relres_plot("log");

}

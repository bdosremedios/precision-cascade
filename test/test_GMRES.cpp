#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

#include "solvers/GMRES.h"

using mxread::MatrixReader;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;

class GMRESTest: public testing::Test {

    protected:
        MatrixReader mr;
        string matrix_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";

    public:
        GMRESTest() {
            mr = MatrixReader();
        }
        ~GMRESTest() = default;

};

TEST_F(GMRESTest, CheckConstruction) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "A_linind_5.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "b_5.csv");
    GMRESSolveTestingMock<double> test_mock(A, b);
    EXPECT_EQ(test_mock.Q_kbasis.rows(), 5);
    EXPECT_EQ(test_mock.Q_kbasis.cols(), 5);
    EXPECT_EQ(test_mock.H.rows(), 6);
    EXPECT_EQ(test_mock.H.cols(), 5);
}

TEST_F(GMRESTest, SolveConvDiff64) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "conv_diff_64_b.csv");
    GMRESSolve<double> gmres_solve_d(A, b);
    // gmres_solve_d.solve();
    // gmres_solve_d.view_relres_plot("log");

}

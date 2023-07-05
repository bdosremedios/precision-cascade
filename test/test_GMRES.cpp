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
        string matrix_dir = "/home/bdosremedios/learn/gmres/test/solve_matrices/";

    public:
        GMRESTest() {
            mr = MatrixReader();
        }
        ~GMRESTest() = default;

};

TEST_F(GMRESTest, SolveSmallerMatrix) {
    
    Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "conv_diff_64_A.csv");
    Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "conv_diff_64_b.csv");
    GMRESSolve<double> gmres_solve_d(A, b);
    // gmres_solve_d.solve();
    // gmres_solve_d.view_relres_plot("log");

}

// #include "gtest/gtest.h"
// #include "Eigen/Dense"
// #include "read_matrix/MatrixReader.h"
// #include <string>

// #include "solvers/Jacobi.h"

// using mxread::MatrixReader;
// using Eigen::MatrixXd;
// using std::string;
// using std::cout, std::endl;

// class JacobiTest: public testing::Test {

//     protected:
//         MatrixReader mr;
//         string matrix_dir = "/home/bdosremedios/learn/gmres/test/solve_matrices/";

//     public:
//         JacobiTest() {
//             mr = MatrixReader();
//         }
//         ~JacobiTest() = default;

// };

// TEST_F(JacobiTest, SolveSmallerMatrix) {
    
//     Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "conv_diff_64_A.csv");
//     Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "conv_diff_64_b.csv");
//     JacobiSolve<double> jacobi_solve_d(A, b);
//     jacobi_solve_d.solve();
//     jacobi_solve_d.view_relres_plot("log");

// }

// TEST_F(JacobiTest, SolveLargerMatrix) {
    
//     Matrix<double, Dynamic, Dynamic> A = mr.read_file_d(matrix_dir + "conv_diff_256_A.csv");
//     Matrix<double, Dynamic, Dynamic> b = mr.read_file_d(matrix_dir + "conv_diff_256_b.csv");
//     JacobiSolve<double> jacobi_solve_d(A, b);
//     jacobi_solve_d.solve();
//     jacobi_solve_d.view_relres_plot("log");
    
// }

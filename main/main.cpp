#include <cmath>
#include <string>
#include <chrono>
#include <iostream>

#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;
using Eigen::MatrixXd, Eigen::MatrixXf;
using std::cout, std::endl;
using std::string;
using std::pow;

int main() {

    // File path variable definitions
    string curr_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
    string A_path = curr_dir + "conv_diff_1024_A.csv";
    string b_path = curr_dir + "conv_diff_1024_b.csv";

    // Calculation variable definitions
    const int n = 1024;
    double convergence_tolerance_single = 1e-4;
    double convergence_tolerance_double = 1e-10;
    float single_tolerance_zero = 4*pow(2, -23);
    double double_tolerance_zero = 4*pow(2, -52);

    // Read matrices
    Matrix<double, Dynamic, Dynamic> A_double = read_matrix_csv<double>(A_path);
    Matrix<double, Dynamic, 1> b_double = read_matrix_csv<double>(b_path);
    Matrix<float, Dynamic, Dynamic> A_single = read_matrix_csv<float>(A_path);
    Matrix<float, Dynamic, 1> b_single = read_matrix_csv<float>(b_path);

    // Initial guess
    Matrix<float, Dynamic, 1> guess = MatrixXf::Random(n, 1);

    // Solve with single to convergence_tolerance_single to get better initial guess
    GMRESSolve<float> gmres_solve_s(
        A_single, b_single, guess, single_tolerance_zero);
    gmres_solve_s.solve(n, convergence_tolerance_single);

    // Solve with double to get better error
    Matrix<float, Dynamic, 1> intermediate_soln = gmres_solve_s.soln();
    Matrix<double, Dynamic, 1> extrap_intermediate_soln = intermediate_soln.cast<double>();

    GMRESSolve<double> gmres_solve_d(
        A_double, b_double, extrap_intermediate_soln, double_tolerance_zero);
    gmres_solve_d.solve(n, convergence_tolerance_double);

    gmres_solve_s.view_relres_plot("log");
    gmres_solve_d.view_relres_plot("log");

    GMRESSolve<double> gmres_straight_solve(
        A_double, b_double, guess.cast<double>(), double_tolerance_zero);
    gmres_straight_solve.solve(n, convergence_tolerance_single);

    gmres_straight_solve.view_relres_plot("log");

    return 0;

}
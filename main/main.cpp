#include <cmath>
#include <string>
#include <chrono>
#include <iostream>

#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

using read_matrix::read_matrix_csv;

using Eigen::MatrixXd, Eigen::MatrixXf;
using MatrixXh = Eigen::Matrix<Eigen::half, Dynamic, Dynamic>;
using Eigen::half;

using std::cout, std::endl;
using std::string;
using std::pow;

int main() {

    // File path variable definitions
    string curr_dir = "/home/bdosremedios/dev/gmres/test/solve_matrices/";
    string A_path = curr_dir + "conv_diff_256_A.csv";
    string b_path = curr_dir + "conv_diff_256_b.csv";

    // Calculation variable definitions
    const int n = 256;
    double convergence_tolerance_single = 1e-4;
    double convergence_tolerance_double = 1e-10;
    half half_tolerance_zero = static_cast<half>(4*pow(2, -10));
    float single_tolerance_zero = 4*pow(2, -23);
    double double_tolerance_zero = 4*pow(2, -52);

    // Read matrices
    Matrix<double, Dynamic, Dynamic> A_double = read_matrix_csv<double>(A_path);
    Matrix<double, Dynamic, 1> b_double = read_matrix_csv<double>(b_path);
    Matrix<float, Dynamic, Dynamic> A_single = read_matrix_csv<float>(A_path);
    Matrix<float, Dynamic, 1> b_single = read_matrix_csv<float>(b_path);
    Matrix<half, Dynamic, Dynamic> A_half = read_matrix_csv<half>(A_path);
    Matrix<half, Dynamic, 1> b_half = read_matrix_csv<half>(b_path);

    // Initial guess
    Matrix<half, Dynamic, 1> guess = MatrixXh::Random(n, 1);

    // Look at solve with each precision
    GMRESSolve<half, half> gmres_solve_h(
        A_half, b_half, guess, half_tolerance_zero
    );
    gmres_solve_h.solve(n, convergence_tolerance_double);

    GMRESSolve<float, float> gmres_solve_s(
        A_single, b_single, guess.cast<float>(), single_tolerance_zero
    );
    gmres_solve_s.solve(n, convergence_tolerance_double);

    GMRESSolve<double, double> gmres_solve_d(
        A_double, b_double, guess.cast<double>(), double_tolerance_zero
    );
    gmres_solve_d.solve(n, convergence_tolerance_double);

    gmres_solve_h.view_relres_plot("log");
    gmres_solve_s.view_relres_plot("log");
    gmres_solve_d.view_relres_plot("log");

    GMRESSolve<half, half> gmres_solve_h_2(
        A_half, b_half, guess, half_tolerance_zero
    );
    gmres_solve_h_2.solve(n, convergence_tolerance_single);

    GMRESSolve<float, float> gmres_solve_s_2(
        A_single, b_single, guess.cast<float>(), single_tolerance_zero
    );
    gmres_solve_s_2.solve(n, convergence_tolerance_single);

    GMRESSolve<double, double> gmres_solve_d_2(
        A_double, b_double, guess.cast<double>(), double_tolerance_zero
    );
    gmres_solve_d_2.solve(n, convergence_tolerance_single);

    gmres_solve_h_2.view_relres_plot("log");
    gmres_solve_s_2.view_relres_plot("log");
    gmres_solve_d_2.view_relres_plot("log");

    return 0;

}
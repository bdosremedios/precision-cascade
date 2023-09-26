#include "Eigen/Dense"
#include "Eigen/SparseCore"

#include "experimentation/SolveRecorder.h"
#include "solvers/krylov/GMRES.h"
#include "tools/MatrixReader.h"

#include <cmath>
#include <memory>
#include <iostream>
#include <string>

using std::pow;
using std::shared_ptr, std::make_shared;

int main() {

    string dev_path = "/home/bdosre/dev/";
    string test_path = dev_path + "numerical_experimentation/test/test_dir_1/test.json";

    string f_A = dev_path + "precision-cascade/test/solve_matrices/conv_diff_256_A.csv";
    string f_b = dev_path + "precision-cascade/test/solve_matrices/conv_diff_256_b.csv";

    shared_ptr<GenericIterativeSolve<MatrixDense>> solver = make_shared<GMRESSolve<MatrixDense, double>>(
        read_matrixCSV<MatrixDense, double>(f_A),
        read_matrixCSV<MatrixVector, double>(f_b),
        pow(2, -52),
        SolveArgPkg()
    );

    solver->solve();

    record_solve(solver, test_path);

    return 0;

}
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/InnerOuterSolve.h"

#include <iostream>

using Eigen::Matrix, Eigen::Dynamic, Eigen::half;

using read_matrix::read_matrix_csv;

using std::cout, std::endl;
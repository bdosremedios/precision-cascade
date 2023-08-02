#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"

#include <string>

using read_matrix::read_matrix_csv;
using Eigen::MatrixXf;
using std::string;
using std::cout, std::endl;
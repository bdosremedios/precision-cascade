#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "../test.h"

#include "read_matrix/MatrixReader.h"
#include "solvers/GMRES.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <memory>
#include <iostream>

using Eigen::Matrix, Eigen::Dynamic;

using read_matrix::read_matrix_csv;

using std::make_shared;
using std::cout, std::endl;


class PGMRESComponentTest: public TestBase {};

#include <cmath>
#include <iostream>

#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include "solvers/Jacobi.h"
#include "solvers/GMRES.h"

using std::cout, std::endl;
using Eigen::half;
using std::pow, std::sqrt;

int main() {

    half a = static_cast<half>(4);
    half b = a*a;
    half c = sqrt(b);
    cout << c << endl;
    
    return 0;

}
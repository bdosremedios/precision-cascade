#include <iostream>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "test.h"

using Eigen::SparseMatrix, Eigen::Matrix;
using Eigen::placeholders::all;
using std::cout, std::endl;

int main() {

    SparseMatrix<double> jam1 = SparseMatrix<double>(5, 5);
    SparseMatrix<Eigen::half> jam2(5, 5);
    jam2.coeffRef(0, 0) = static_cast<Eigen::half>(1);
    jam2.coeffRef(1, 1) = static_cast<Eigen::half>(2);
    jam2.coeffRef(2, 2) = static_cast<Eigen::half>(3);
    jam2.coeffRef(3, 3) = static_cast<Eigen::half>(4);
    jam2.coeffRef(4, 4) = static_cast<Eigen::half>(5);
    jam2.coeffRef(4, 1) = static_cast<Eigen::half>(5);
    Matrix<Eigen::half, 5, 1> vec{static_cast<Eigen::half>(1),
                                  static_cast<Eigen::half>(2),
                                  static_cast<Eigen::half>(3),
                                  static_cast<Eigen::half>(4),
                                  static_cast<Eigen::half>(5)};

    cout << jam2.cols() << endl;
    cout << jam2.rows() << endl;
    cout << static_cast<Eigen::half>(4)*(jam2*vec) << endl;

    vec = jam2.col(1);
    cout << vec << endl << endl;
    cout << jam2 << endl << endl;

    cout << jam2.transpose() << endl;

    cout << jam2.block(0, 1, 3, 3) << endl;

    cout << jam2.col(1).norm() << endl;

    Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jam3 = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(5, 5);
    cout << jam3.col(3) << endl;
    cout << jam3.coeffRef(2, 1) << endl;

    cout << jam2.coeff(1, 1) << endl;
    cout << jam3.coeff(1, 1) << endl;

    DenseMatrix<double> jam4(5, 5);

    cout << jam4.coeff(1, 1) << endl;
    cout << jam4.coeffRef(1, 1) << endl;
    // cout << jam4.coeff(6, 6) << endl << endl;
    // cout << jam4.coeffRef(6, 6) << endl << endl;

    cout << jam4 << endl << endl;
    jam4.coeffRef(1, 1) = 1;
    cout << jam4 << endl;

    // cout << jam2.coeff(6, 6) << endl;

    return 0;

}
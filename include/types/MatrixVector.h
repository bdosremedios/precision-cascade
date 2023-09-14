#ifndef VECTOR_H
#define VECTOR_H

#include "Eigen/Dense"

using Eigen::Matrix;
using Eigen::Dynamic;

template <typename T>
using MatrixVector = Matrix<T, Dynamic, 1>;

#endif
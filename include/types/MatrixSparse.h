#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "Eigen/SparseCore"

using Eigen::SparseMatrix;

template <typename T>
using MatrixSparse = SparseMatrix<T>;

#endif
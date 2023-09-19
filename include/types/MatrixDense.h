#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include "Eigen/Dense"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Index;

template <typename T>
class MatrixDense: public Matrix<T, Dynamic, Dynamic>
{
public:

    const T& coeff(Index row, Index col) const {
        return this->operator()(row, col);
    }

    T& coeffRef(Index row, Index col) {
        return this->operator()(row, col);
    }
    
    using Matrix<T, Dynamic, Dynamic>::Matrix;

};

#endif
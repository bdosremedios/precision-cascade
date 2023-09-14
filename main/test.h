#ifndef TEST_H
#define TEST_H

#include "Eigen/Dense"
#include "Eigen/SparseCore"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Index;

template <typename T>
class DenseMatrix: public Matrix<T, Dynamic, Dynamic> {
    
    public:

        const T& coeff(Index row, Index col) const {
            return this->operator()(row, col);
        }

        T& coeffRef(Index row, Index col) {
            return this->operator()(row, col);
        }
        
        using Matrix<T, Dynamic, Dynamic>::Matrix;

};

// template <typename M, typename T>
// class MatrixAdapter {

//     public:
        
//         M mat;

//         MatrixAdapter(M arg_mat): mat(arg_mat) {}

//         Scalar access(Index row, Index col) const = 0;

//         Scalar &accessRef(Index row, Index col) = 0;

//         int cols() { return mat.cols(); }
//         int rows() { return mat.rows(); }

//         M block(Index   startRow,
//                 Index 	startCol,
//                 Index 	blockRows,
//                 Index 	blockCols ) {

//         }

// };





#endif
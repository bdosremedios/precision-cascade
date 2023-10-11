#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "Eigen/SparseCore"

#include <cmath>

using Eigen::Matrix;
using Eigen::SparseMatrix;
using Eigen::Dynamic;
using Eigen::Index;

using std::min;

template <typename T>
class MatrixSparse: public SparseMatrix<T>
{
public:
    
    static SparseMatrix<T> Random(Index m, Index n) {
        return Matrix<T, Dynamic, Dynamic>::Random(m, n).sparseView();
    }
    
    static SparseMatrix<T> Identity(Index m, Index n) {
        SparseMatrix<T> temp = SparseMatrix<T>(m, n);
        for (int i=0; i<min(m, n); ++i) { temp.coeffRef(i, i) = static_cast<T>(1); }
        return temp;
    }

    static SparseMatrix<T> Ones(Index m, Index n) {
        return Matrix<T, Dynamic, Dynamic>::Ones(m, n).sparseView();
    }

    static SparseMatrix<T> Zero(Index m, Index n) {
        return SparseMatrix<T>(m, n);
    }
    
    using SparseMatrix<T>::SparseMatrix;

    bool operator==(const MatrixSparse<T> &rhs) const {
        return this->isApprox(rhs);
    }

    void reduce() { this->prune(static_cast<T>(0.)); }

};

#endif
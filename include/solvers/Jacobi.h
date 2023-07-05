#ifndef JACOBI_H
#define JACOBI_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout, std::endl;
using Eigen::DiagonalMatrix;

template <typename T>
class JacobiSolve: public LinearSolve<T> {

    protected:
        Matrix<T, Dynamic, 1> iterate() const override {

            Matrix<T, Dynamic, 1> A_diag = this->A.diagonal();
            Matrix<T, Dynamic, Dynamic> invD = DiagonalMatrix<T, Dynamic>(A_diag.array().inverse());

            return this->x + invD*(this->b-this->A*this->x);

        }
    
    public:
        // Inherit constructors
        using LinearSolve<T>::LinearSolve;

};

#endif
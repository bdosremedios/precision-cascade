#ifndef JACOBI_H
#define JACOBI_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout, std::endl;
using Eigen::Matrix;

template <typename T>
class JacobiSolve: public LinearSolve<T> {

    protected:

        void iterate() override {
            
            for (int i=0; i<this->m; ++i) {
                Matrix<T, Dynamic, 1> x_temp = this->x;
                T acc = this->b(i);
                for (int j=0; j<this->m; ++j) {
                    acc -= this->A(i, j)*x_temp(j);
                }
                this->x(i) += (static_cast<T>(1)/(this->A(i, i)))*acc;
            }

        }
    
    public:

        // Inherit constructors
        using LinearSolve<T>::LinearSolve;

};

#endif
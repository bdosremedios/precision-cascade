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

        using LinearSolve<T>::m;
        using LinearSolve<T>::A;
        using LinearSolve<T>::b;
        using LinearSolve<T>::x;

        void iterate() override {
            
            for (int i=0; i<m; ++i) {

                Matrix<T, Dynamic, 1> x_temp = x;

                T acc = b(i);
                for (int j=0; j<m; ++j) {
                    acc -= A(i, j)*x_temp(j);
                }

                x(i) += (static_cast<T>(1)/(A(i, i)))*acc;

            }

        }
    
    public:

        // Inherit constructors
        using LinearSolve<T>::LinearSolve;

};

#endif
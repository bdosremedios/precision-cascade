#ifndef JACOBI_H
#define JACOBI_H

#include "Eigen/Dense"

#include <iostream>

#include "LinearSolve.h"

using std::cout, std::endl;
using Eigen::Matrix;

template <typename T>
class JacobiSolve: public LinearSolve<T> {

    protected:

        using LinearSolve<T>::m;
        using LinearSolve<T>::A;
        using LinearSolve<T>::b;
        using LinearSolve<T>::x;

        // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

        void iterate() override {

            Matrix<T, Dynamic, 1> x_temp = x;
            
            for (int i=0; i<m; ++i) {

                T acc = b(i);
                for (int j=0; j<m; ++j) {
                    acc -= A(i, j)*x_temp(j);
                }

                x(i) = x_temp(i) + acc/(A(i, i));

            }

        }

        void derived_reset() override {}; // Set reset as empty function
    
    public:

        // *** CONSTRUCTORS ***

        using LinearSolve<T>::LinearSolve;


};

#endif
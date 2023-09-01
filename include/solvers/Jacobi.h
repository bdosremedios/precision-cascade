#ifndef JACOBI_H
#define JACOBI_H

#include "Eigen/Dense"

#include <iostream>

#include "IterativeSolve.h"

using std::cout, std::endl;
using Eigen::Matrix;

template <typename T>
class JacobiSolve: public TypedIterativeSolve<T> {

    protected:

        using TypedIterativeSolve<T>::m;
        using TypedIterativeSolve<T>::A;
        using TypedIterativeSolve<T>::b;
        using TypedIterativeSolve<T>::typed_soln;

        // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

        void iterate() override {

            Matrix<T, Dynamic, 1> prev_soln = typed_soln;
            
            for (int i=0; i<m; ++i) {

                T acc = b(i);
                for (int j=0; j<m; ++j) {
                    acc -= A(i, j)*prev_soln(j);
                }

                typed_soln(i) = prev_soln(i) + acc/(A(i, i));

            }

        }

        void derived_reset() override {}; // Set reset as empty function
    
    public:

        // *** CONSTRUCTORS ***

        using TypedIterativeSolve<T>::TypedIterativeSolve;


};

#endif
#ifndef GMRES_H
#define GMRES_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout, std::endl;

template <typename T>
class GMRESSolve: public LinearSolve<T> {

    protected:
        Matrix<T, Dynamic, Dynamic> krylovBasis;

    protected:
        Matrix<T, Dynamic, 1> iterate() const override {

            return Matrix<T, Dynamic, 1>::Ones(1, 1);

        }
    
    public:
        // Constructors
        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                    const Matrix<T, Dynamic, 1> arg_b): LinearSolve<T>::LinearSolve(arg_A, arg_b) {
            constructorHelper();
        }

        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                    const Matrix<T, Dynamic, 1> arg_b, 
                    const Matrix<T, Dynamic, 1> arg_x_0): LinearSolve<T>::LinearSolve(arg_A, arg_b) {
            constructorHelper();
        }

        void constructorHelper() {
            krylovBasis = Matrix<T, Dynamic, Dynamic>();
            krylovBasis.resize(this->m, this->n);
        }

};

#endif
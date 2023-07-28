#ifndef SOR_H
#define SOR_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout, std::endl;
using Eigen::Matrix;

template <typename T>
class SORSolve: public LinearSolve<T> {

    protected:

        using LinearSolve<T>::m;
        using LinearSolve<T>::A;
        using LinearSolve<T>::b;
        using LinearSolve<T>::x;

        T w;

        void iterate() override {

            Matrix<T, Dynamic, 1> x_k = x;
            
            for (int i=0; i<m; ++i) {

                T acc = b(i);
                for (int j=i+1; j<m; ++j) {
                    acc -= A(i, j)*x_k(j);
                }
                for (int j=0; j<i; ++j) {
                    acc -= A(i, j)*x(j);
                }

                x(i) = (static_cast<T>(1)-w)*x_k(i) + w*acc/(A(i, i));

            }

        }
    
    public:

        // Constructors
        SORSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                 const Matrix<T, Dynamic, 1> arg_b,
                   T arg_w):
            w(arg_w), LinearSolve<T>::LinearSolve(arg_A, arg_b) {}

        SORSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                 const Matrix<T, Dynamic, 1> arg_b, 
                 const Matrix<T, Dynamic, 1> arg_x_0,
                 T arg_w):
            w(arg_w), LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0) {}

};

#endif
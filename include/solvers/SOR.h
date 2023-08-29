#ifndef SOR_H
#define SOR_H

#include "Eigen/Dense"

#include <iostream>

#include "IterativeSolve.h"

using std::cout, std::endl;
using Eigen::Matrix, Eigen::Dynamic;

template <typename T>
class SORSolve: public TypedIterativeSolve<T> {

    protected:

        using TypedIterativeSolve<T>::m;
        using TypedIterativeSolve<T>::A;
        using TypedIterativeSolve<T>::b;
        using TypedIterativeSolve<T>::x;

        T w;

        // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

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

        void derived_reset() override {}; // Set reset as empty function
    
    public:

        // *** CONSTRUCTORS ***

        SORSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_w,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            SORSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_w,
                arg_max_outer_iter, arg_target_rel_res
            )
        {}

        SORSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b, 
            Matrix<T, Dynamic, 1> const &arg_x_0,
            T const &arg_w,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            w(arg_w),
            TypedIterativeSolve<T>::TypedIterativeSolve(arg_A, arg_b, arg_x_0, arg_max_outer_iter, arg_target_rel_res)
        {}

};

#endif
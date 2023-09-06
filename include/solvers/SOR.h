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
        using TypedIterativeSolve<T>::A_T;
        using TypedIterativeSolve<T>::b_T;
        using TypedIterativeSolve<T>::typed_soln;

        T w;

        // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

        void typed_iterate() override {

            Matrix<T, Dynamic, 1> prev_soln = typed_soln;
            
            for (int i=0; i<m; ++i) {

                T acc = b_T(i);
                for (int j=i+1; j<m; ++j) {
                    acc -= A_T(i, j)*prev_soln(j);
                }
                for (int j=0; j<i; ++j) {
                    acc -= A_T(i, j)*typed_soln(j);
                }

                typed_soln(i) = (static_cast<T>(1)-w)*prev_soln(i) + w*acc/(A_T(i, i));

            }

        }

        void derived_typed_reset() override {}; // Set reset as empty function
    
    public:

        // *** CONSTRUCTORS ***

        SORSolve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            double const &arg_w,
            int const &arg_max_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            SORSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_w,
                arg_max_iter, arg_target_rel_res
            )
        {}

        SORSolve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b, 
            Matrix<double, Dynamic, 1> const &arg_x_0,
            double const &arg_w,
            int const &arg_max_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            w(static_cast<T>(arg_w)),
            TypedIterativeSolve<T>::TypedIterativeSolve(
                arg_A, arg_b, arg_x_0, arg_max_iter, arg_target_rel_res
            )
        {}

};

#endif
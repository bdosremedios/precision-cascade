#ifndef SOR_H
#define SOR_H

#include "../IterativeSolve.h"

template <typename T>
class SORSolve: public TypedIterativeSolve<T>
{
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
        SolveArgPkg const &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<T>::TypedIterativeSolve(arg_A, arg_b, arg_pkg)
    {}

};

#endif
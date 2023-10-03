#ifndef SOR_H
#define SOR_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class SORSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::typed_soln;

    T w;

    // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

    void typed_iterate() override {

        MatrixVector<T> prev_soln = typed_soln;
        for (int i=0; i < typed_lin_sys.get_m(); ++i) {

            T acc = typed_lin_sys.get_b_typed()(i);

            for (int j=i+1; j < typed_lin_sys.get_m(); ++j) {
                acc -= typed_lin_sys.get_A_typed().coeff(i, j)*prev_soln(j);
            }

            for (int j=0; j < i; ++j) {
                acc -= typed_lin_sys.get_A_typed().coeff(i, j)*typed_soln(j);
            }

            typed_soln(i) = (static_cast<T>(1)-w)*prev_soln(i) +
                            w*acc/(typed_lin_sys.get_A_typed().coeff(i, i));

        }

    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** CONSTRUCTORS ***

    SORSolve(
        const TypedLinearSystem<M, T> &arg_typed_lin_sys,
        const double &arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys, arg_pkg)
    {}

};

#endif
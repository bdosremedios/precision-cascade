#ifndef JACOBI_H
#define JACOBI_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class JacobiSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::typed_soln;

    // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

    void typed_iterate() override {

        MatrixVector<T> prev_soln = typed_soln;
        for (int i=0; i < typed_lin_sys.get_m(); ++i) {
            T acc = typed_lin_sys.get_b_typed()(i);
            for (int j=0; j < typed_lin_sys.get_m(); ++j) {
                acc -= typed_lin_sys.get_A_typed().coeff(i, j)*prev_soln(j);
            }
            typed_soln(i) = prev_soln(i) + acc/(typed_lin_sys.get_A_typed().coeff(i, i));
        }

    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** CONSTRUCTORS ***

    using TypedIterativeSolve<M, T>::TypedIterativeSolve;

};

#endif
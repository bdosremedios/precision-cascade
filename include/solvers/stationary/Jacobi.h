#ifndef JACOBI_H
#define JACOBI_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class JacobiSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::m;
    using TypedIterativeSolve<M, T>::A_T;
    using TypedIterativeSolve<M, T>::b_T;
    using TypedIterativeSolve<M, T>::typed_soln;

    // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

    void typed_iterate() override {

        MatrixVector<T> prev_soln = typed_soln;
        for (int i=0; i<m; ++i) {
            T acc = b_T(i);
            for (int j=0; j<m; ++j) { acc -= A_T.coeff(i, j)*prev_soln(j); }
            typed_soln(i) = prev_soln(i) + acc/(A_T.coeff(i, i));
        }

    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** CONSTRUCTORS ***

    using TypedIterativeSolve<M, T>::TypedIterativeSolve;

};

#endif
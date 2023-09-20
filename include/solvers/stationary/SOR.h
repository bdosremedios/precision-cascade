#ifndef SOR_H
#define SOR_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class SORSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::m;
    using TypedIterativeSolve<M, T>::A_T;
    using TypedIterativeSolve<M, T>::b_T;
    using TypedIterativeSolve<M, T>::typed_soln;

    T w;

    // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

    void typed_iterate() override {

        MatrixVector<T> prev_soln = typed_soln;
        for (int i=0; i<m; ++i) {
            T acc = b_T(i);
            for (int j=i+1; j<m; ++j) { acc -= A_T.coeff(i, j)*prev_soln(j); }
            for (int j=0; j<i; ++j) { acc -= A_T.coeff(i, j)*typed_soln(j); }
            typed_soln(i) = (static_cast<T>(1)-w)*prev_soln(i) + w*acc/(A_T.coeff(i, i));
        }

    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** CONSTRUCTORS ***

    SORSolve(
        M<double> const &arg_A,
        MatrixVector<double> const &arg_b,
        double const &arg_w,
        SolveArgPkg const &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_A, arg_b, arg_pkg)
    {}

};

#endif
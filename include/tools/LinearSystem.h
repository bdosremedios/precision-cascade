#ifndef LINEARSYSTEM_H
#define LINEARSYSTEM_H

#include "types/types.h"

template <template <typename> typename M>
class LinearSystem {
    
    const M<double> A;
    const MatrixVector<double> b;

    LinearSystem(
        M<double> arg_A, MatrixVector<double> arg_b
    ):
        A(arg_A),
        b(arg_b)
    {}

    M<double> &A() const { return A; }

    MatrixVector<double> &b() const { return b; }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public LinearSystem {

    const M<T> A_Typed;
    const MatrixVector<T> b_Typed;

    LinearSystem(
        M<double> arg_A, MatrixVector<double> arg_b
    ):
        A_Typed(arg_A.template cast<T>()),
        b_Typed(arg_b.template cast<T>()),
        LinearSystem(arg_A, arg_b)
    {}

    M<T> &A_Typed() const { return A_Typed; }

    MatrixVector<T> &b_Typed() const { return b_Typed; }

}

#endif
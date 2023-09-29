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

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public LinearSystem {
    
    const M<T> A_T;
    const MatrixVector<T> b_T;

    LinearSystem(
        M<double> arg_A, MatrixVector<double> arg_b
    ):
        A_T(arg_A.template cast<T>()),
        b_T(arg_b.template cast<T>()),
        LinearSystem(arg_A, arg_b)
    {}

}

#endif
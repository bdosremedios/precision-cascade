#ifndef LINEARSYSTEM_H
#define LINEARSYSTEM_H

#include "types/types.h"

template <template <typename> typename M>
class GenericLinearSystem
{
protected:

    const int m;
    const int n;
    const M<double> A;
    const MatrixVector<double> b;

public:

    GenericLinearSystem(
        M<double> arg_A,
        MatrixVector<double> arg_b
    ):
        m(arg_A.rows()),
        n(arg_A.cols()),
        A(arg_A),
        b(arg_b)
    {
        if ((m < 1) || (n < 1)) { throw runtime_error("Empty Matrix A"); }
        if (m != b.rows()) { throw runtime_error("A not compatible with b for linear system"); }
    }

    const M<double> &get_A() const { return A; }

    const MatrixVector<double> &get_b() const { return b; }

    const int &get_m() const { return m; }

    const int &get_n() const { return n; }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public GenericLinearSystem<M>
{
protected:

    const M<T> A_typed;
    const MatrixVector<T> b_typed;

public:

    TypedLinearSystem(
        M<double> arg_A,
        MatrixVector<double> arg_b
    ):
        A_typed(arg_A.template cast<T>()),
        b_typed(arg_b.template cast<T>()),
        GenericLinearSystem<M>(arg_A, arg_b)
    {}

    const M<T> &get_A_typed() const { return A_typed; }

    const MatrixVector<T> &get_b_typed() const { return b_typed; }

};

#endif
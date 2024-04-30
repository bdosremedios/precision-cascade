#ifndef LINEARSYSTEM_H
#define LINEARSYSTEM_H

#include "tools/cuHandleBundle.h"

#include "types/types.h"

template <template <typename> typename M>
class GenericLinearSystem
{
protected:

    const int m;
    const int n;
    const int nnz;
    const cuHandleBundle cu_handles;
    const M<double> A;
    const Vector<double> b;

public:

    // *** Constructors ***
    GenericLinearSystem(
        M<double> arg_A,
        Vector<double> arg_b
    ):
        m(arg_A.rows()),
        n(arg_A.cols()),
        nnz(arg_A.non_zeros()),
        cu_handles(arg_A.get_cu_handles()),
        A(arg_A),
        b(arg_b)
    {
        if ((m < 1) || (n < 1)) {
            throw std::runtime_error("GenericLinearSystem: Empty Matrix A");
        }
        if (m != b.rows()) {
            throw std::runtime_error("GenericLinearSystem: A not compatible with b for linear system");
        }
    }

    // *** Getters ***
    const M<double> &get_A() const { return A; }
    virtual const Vector<double> &get_b() const { return b; }
    const int &get_m() const { return m; }
    const int &get_n() const { return n; }
    const int &get_nnz() const { return nnz; }
    const cuHandleBundle &get_cu_handles() const { return cu_handles; }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public GenericLinearSystem<M>
{
private:

    const M<T> A_typed;
    const Vector<T> b_typed;

public:

    // *** Constructors ***
    TypedLinearSystem(
        M<double> arg_A,
        Vector<double> arg_b
    ):
        A_typed(arg_A.template cast<T>()),
        b_typed(arg_b.template cast<T>()),
        GenericLinearSystem<M>(arg_A, arg_b)
    {}

    // *** Getters ***
    const M<T> &get_A_typed() const { return A_typed; }
    virtual const Vector<T> &get_b_typed() const { return b_typed; }

};

template <template <typename> typename M, typename T>
class Mutb_TypedLinearSystem: public TypedLinearSystem<M, T>
{
private:
    
    Vector<double> b;
    Vector<T> b_typed;

public:

    // *** Constructors ***
    Mutb_TypedLinearSystem(
        M<double> arg_A,
        Vector<double> arg_b
    ):
        b(arg_b),
        b_typed(arg_b.template cast<T>()),
        TypedLinearSystem<M, T>(arg_A, arg_b)
    {}

    // *** Getters ***
    const Vector<double> &get_b() const override { return b; }
    const Vector<T> &get_b_typed() const override { return b_typed; }

    // *** Setters ***
    void set_b(Vector<double> arg_b) {
        b = arg_b;
        b_typed = b.template cast<T>();
    }

};

#endif
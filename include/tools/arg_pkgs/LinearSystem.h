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
    GenericLinearSystem(const M<double> &arg_A, const Vector<double> &arg_b):
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

public:

    // *** Getters ***
    const M<double> &get_A() const { return A; }
    const Vector<double> &get_b() const { return b; }
    const int get_m() const { return m; }
    const int get_n() const { return n; }
    const int get_nnz() const { return nnz; }
    const cuHandleBundle get_cu_handles() const { return cu_handles; }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem_Intf
{
protected:

    const GenericLinearSystem<M> * const gen_lin_sys_ptr;

public:

    TypedLinearSystem_Intf(const GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        gen_lin_sys_ptr(arg_gen_lin_sys_ptr)
    {}

    const GenericLinearSystem<M> * get_gen_lin_sys_ptr() const { return gen_lin_sys_ptr; }

    // Wrapper behavior, wrapping GenericLinearSystem<M> methods
    const M<double> &get_A() const { return gen_lin_sys_ptr->get_A(); }
    virtual const Vector<double> &get_b() const { return gen_lin_sys_ptr->get_b(); }
    const int get_m() const { return gen_lin_sys_ptr->get_m(); }
    const int get_n() const { return gen_lin_sys_ptr->get_n(); }
    const int get_nnz() const { return gen_lin_sys_ptr->get_nnz(); }
    const cuHandleBundle get_cu_handles() const { return gen_lin_sys_ptr->get_cu_handles(); }

    // *** Pure virtual methods ***
    virtual const M<T> &get_A_typed() const = 0;
    virtual const Vector<T> &get_b_typed() const = 0;

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public TypedLinearSystem_Intf<M, T>
{
private:

    const M<T> A_typed;
    const Vector<T> b_typed;

public:

    TypedLinearSystem(const GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        TypedLinearSystem_Intf<M, T>(arg_gen_lin_sys_ptr),
        A_typed(this->gen_lin_sys_ptr->get_A().template cast<T>()),
        b_typed(this->gen_lin_sys_ptr->get_b().template cast<T>())
    {}

    // *** Implemented virtual methods ***
    const M<T> &get_A_typed() const override { return A_typed; }
    const Vector<T> &get_b_typed() const override { return b_typed; }

};

// Partial specialization with double to remove unneccessary repeated data of typed vector/matrix
template <template <typename> typename M>
class TypedLinearSystem<M, double>: public TypedLinearSystem_Intf<M, double>
{
public:

    TypedLinearSystem(const GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        TypedLinearSystem_Intf<M, double>(arg_gen_lin_sys_ptr)
    {}

    // *** Implemented virtual methods ***
    const M<double> &get_A_typed() const override { return this->gen_lin_sys_ptr->get_A(); }
    const Vector<double> &get_b_typed() const override { return this->gen_lin_sys_ptr->get_b(); }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem_MutAddlRHS: public TypedLinearSystem_Intf<M, T>
{
private:

    Vector<double> additional_rhs;
    Vector<T> additional_rhs_typed;
    const TypedLinearSystem<M, T> * const orig_typed_lin_sys_ptr;

public:

    TypedLinearSystem_MutAddlRHS(
        const TypedLinearSystem<M, T> * const arg_orig_typed_lin_sys_ptr,
        const Vector<double> &arg_additional_rhs
    ):
        TypedLinearSystem_Intf<M, T>(arg_orig_typed_lin_sys_ptr->get_gen_lin_sys_ptr()),
        orig_typed_lin_sys_ptr(arg_orig_typed_lin_sys_ptr),
        additional_rhs(arg_additional_rhs),
        additional_rhs_typed(arg_additional_rhs.template cast<T>())
    {}

    // *** Implemented virtual methods ***
    const Vector<double> &get_b() const override { return additional_rhs; }
    const M<T> &get_A_typed() const override { return orig_typed_lin_sys_ptr->get_A_typed(); }
    const Vector<T> &get_b_typed() const override { return additional_rhs_typed; }

    void set_rhs(const Vector<double> &arg_rhs) {
        if (orig_typed_lin_sys_ptr->get_m() != arg_rhs.rows()) {
            throw std::runtime_error(
                "TypedLinearSystem_MutableAdditionalRHS: rhs for linear system in set_rhs is incompatible"
            );
        }
        additional_rhs = arg_rhs;
        additional_rhs_typed = arg_rhs.template cast<T>();
    }

};

#endif
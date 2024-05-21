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
    Vector<double> b;

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

    // *** Setters ***
    void set_b(const Vector<double> &arg_b) {
        if (m != arg_b.rows()) {
            throw std::runtime_error("GenericLinearSystem: b for linear system in set_b");
        }
        b = arg_b;
    }

};

template <template <typename> typename M, typename T>
class TypedLinearSystem_Base
{
protected:

    GenericLinearSystem<M> * const gen_lin_sys_ptr;

public:

    TypedLinearSystem_Base(GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        gen_lin_sys_ptr(arg_gen_lin_sys_ptr)
    {}

    GenericLinearSystem<M> * get_gen_lin_sys_ptr() const { return gen_lin_sys_ptr; }

    // Wrapper behavior, wrapping GenericLinearSystem<M> methods
    const M<double> &get_A() const { return gen_lin_sys_ptr->get_A(); }
    const Vector<double> &get_b() const { return gen_lin_sys_ptr->get_b(); }
    const int get_m() const { return gen_lin_sys_ptr->get_m(); }
    const int get_n() const { return gen_lin_sys_ptr->get_n(); }
    const int get_nnz() const { return gen_lin_sys_ptr->get_nnz(); }
    const cuHandleBundle get_cu_handles() const { return gen_lin_sys_ptr->get_cu_handles(); }

    // *** Pure virtual methods ***
    virtual const M<T> &get_A_typed() const = 0;
    virtual const Vector<T> &get_b_typed() const = 0;
    virtual void set_b(const Vector<double> &arg_b) = 0;

};

template <template <typename> typename M, typename T>
class TypedLinearSystem: public TypedLinearSystem_Base<M, T>
{
private:

    const M<T> A_typed;
    Vector<T> b_typed;

public:

    TypedLinearSystem(GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        TypedLinearSystem_Base<M, T>(arg_gen_lin_sys_ptr),
        A_typed(this->gen_lin_sys_ptr->get_A().template cast<T>()),
        b_typed(this->gen_lin_sys_ptr->get_b().template cast<T>())
    {}

    // *** Implemented virtual methods ***
    const M<T> &get_A_typed() const override { return A_typed; }
    const Vector<T> &get_b_typed() const override { return b_typed; }
    void set_b(const Vector<double> &arg_b) override {
        this->gen_lin_sys_ptr->set_b(arg_b);
        b_typed = this->gen_lin_sys_ptr->get_b().template cast<T>();
    }

};

// Partial specialization with double to remove unneccessary repeated data of typed vector/matrix
template <template <typename> typename M>
class TypedLinearSystem<M, double>: public TypedLinearSystem_Base<M, double>
{
public:

    TypedLinearSystem(GenericLinearSystem<M> * const arg_gen_lin_sys_ptr):
        TypedLinearSystem_Base<M, double>(arg_gen_lin_sys_ptr)
    {}

    // *** Implemented virtual methods ***
    const M<double> &get_A_typed() const override { return this->gen_lin_sys_ptr->get_A(); }
    const Vector<double> &get_b_typed() const override { return this->gen_lin_sys_ptr->get_b(); }
    void set_b(const Vector<double> &arg_b) override { this->gen_lin_sys_ptr->set_b(arg_b); }

};

#endif
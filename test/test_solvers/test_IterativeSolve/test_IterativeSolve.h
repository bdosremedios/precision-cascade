#ifndef TEST_ITERATIVESOLVE_H
#define TEST_ITERATIVESOLVE_H

#include "solvers/IterativeSolve.h"

template <template<typename> typename M, typename T>
class TypedIterativeSolveTestingMock: public TypedIterativeSolve<M, T> {

    void typed_iterate() override { typed_soln = soln; }
    void derived_typed_reset() override {}

    public:

        Vector<T> soln;

        using TypedIterativeSolve<M, T>::gen_lin_sys_ptr;
        using TypedIterativeSolve<M, T>::init_guess;
        using TypedIterativeSolve<M, T>::generic_soln;
    
        using TypedIterativeSolve<M, T>::typed_lin_sys_ptr;
        using TypedIterativeSolve<M, T>::init_guess_typed;
        using TypedIterativeSolve<M, T>::typed_soln;

        using TypedIterativeSolve<M, T>::initiated;
        using TypedIterativeSolve<M, T>::converged;
        using TypedIterativeSolve<M, T>::terminated;
        using TypedIterativeSolve<M, T>::curr_iter;
        using TypedIterativeSolve<M, T>::max_iter;
        using TypedIterativeSolve<M, T>::target_rel_res;
        using TypedIterativeSolve<M, T>::res_norm_history;

        TypedIterativeSolveTestingMock(
            TypedLinearSystem<M, T> * const arg_typed_lin_sys_ptr,
            const Vector<T> &arg_soln_typed,
            const SolveArgPkg &arg_pkg
        ):
            soln(arg_soln_typed),
            TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, arg_pkg)
        {}

};

#endif
#ifndef TEST_ITERATIVESOLVE_H
#define TEST_ITERATIVESOLVE_H

#include "solvers/IterativeSolve.h"

template <template<typename> typename TMatrix, typename TPrecision>
class TypedIterativeSolveTestingMock:
    public TypedIterativeSolve<TMatrix, TPrecision>
{
protected:

    void typed_iterate() override { this->typed_soln = soln; }
    void derived_typed_reset() override {}

public:

    Vector<TPrecision> soln;

    using TypedIterativeSolve<TMatrix, TPrecision>::gen_lin_sys_ptr;
    using TypedIterativeSolve<TMatrix, TPrecision>::init_guess;

    using TypedIterativeSolve<TMatrix, TPrecision>::typed_lin_sys_ptr;
    using TypedIterativeSolve<TMatrix, TPrecision>::init_guess_typed;

    using TypedIterativeSolve<TMatrix, TPrecision>::max_iter;
    using TypedIterativeSolve<TMatrix, TPrecision>::target_rel_res;

    TypedIterativeSolveTestingMock(
        TypedLinearSystem<TMatrix, TPrecision> * const arg_typed_lin_sys_ptr,
        const Vector<TPrecision> &arg_soln_typed,
        const SolveArgPkg &arg_pkg
    ):
        soln(arg_soln_typed),
        TypedIterativeSolve<TMatrix, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr,
            arg_pkg
        )
    {}

};

#endif
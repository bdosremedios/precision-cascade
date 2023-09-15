#ifndef TEST_ITERATIVESOLVE_H
#define TEST_ITERATIVESOLVE_H

#include "solvers/IterativeSolve.h"

template <template<typename> typename M, typename T>
class TypedIterativeSolveTestingMock: public TypedIterativeSolve<M, T> {

    void typed_iterate() override { typed_soln = soln; }
    void derived_typed_reset() override {}

    public:

        MatrixVector<T> soln;

        using TypedIterativeSolve<M, T>::m;
        using TypedIterativeSolve<M, T>::n;

        using TypedIterativeSolve<M, T>::A;
        using TypedIterativeSolve<M, T>::b;
        using TypedIterativeSolve<M, T>::init_guess;
        using TypedIterativeSolve<M, T>::generic_soln;
    
        using TypedIterativeSolve<M, T>::A_T;
        using TypedIterativeSolve<M, T>::b_T;
        using TypedIterativeSolve<M, T>::init_guess_T;
        using TypedIterativeSolve<M, T>::typed_soln;

        using TypedIterativeSolve<M, T>::initiated;
        using TypedIterativeSolve<M, T>::converged;
        using TypedIterativeSolve<M, T>::terminated;
        using TypedIterativeSolve<M, T>::curr_iter;
        using TypedIterativeSolve<M, T>::max_iter;
        using TypedIterativeSolve<M, T>::target_rel_res;
        using TypedIterativeSolve<M, T>::res_hist;
        using TypedIterativeSolve<M, T>::res_norm_hist;

        TypedIterativeSolveTestingMock(
            M<double> const &arg_A,
            MatrixVector<double> const &arg_b,
            MatrixVector<T> const &arg_soln,
            SolveArgPkg const &arg_pkg
        ):
            soln(arg_soln),
            TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_A, arg_b, arg_pkg)
        {}

};

#endif
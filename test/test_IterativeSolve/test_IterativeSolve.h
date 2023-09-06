#ifndef TEST_ITERATIVESOLVE_H
#define TEST_ITERATIVESOLVE_H

#include "solvers/IterativeSolve.h"

template <typename T>
class TypedIterativeSolveTestingMock: public TypedIterativeSolve<T> {

    void typed_iterate() override { typed_soln = soln; }
    void derived_typed_reset() override {}

    public:

        Matrix<T, Dynamic, 1> soln;

        using TypedIterativeSolve<T>::m;
        using TypedIterativeSolve<T>::n;

        using TypedIterativeSolve<T>::A;
        using TypedIterativeSolve<T>::b;
        using TypedIterativeSolve<T>::init_guess;
        using TypedIterativeSolve<T>::generic_soln;
    
        using TypedIterativeSolve<T>::A_T;
        using TypedIterativeSolve<T>::b_T;
        using TypedIterativeSolve<T>::init_guess_T;
        using TypedIterativeSolve<T>::typed_soln;

        using TypedIterativeSolve<T>::initiated;
        using TypedIterativeSolve<T>::converged;
        using TypedIterativeSolve<T>::terminated;
        using TypedIterativeSolve<T>::curr_outer_iter;
        using TypedIterativeSolve<T>::max_outer_iter;
        using TypedIterativeSolve<T>::target_rel_res;
        using TypedIterativeSolve<T>::res_hist;
        using TypedIterativeSolve<T>::res_norm_hist;

        TypedIterativeSolveTestingMock(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_soln,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            soln(arg_soln),
            TypedIterativeSolve<T>::TypedIterativeSolve(
                arg_A, arg_b, arg_max_outer_iter, arg_target_rel_res
            )
        {}

        TypedIterativeSolveTestingMock(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            Matrix<T, Dynamic, 1> const &arg_soln,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            soln(arg_soln),
            TypedIterativeSolve<T>::TypedIterativeSolve(
                arg_A, arg_b, arg_init_guess, arg_max_outer_iter, arg_target_rel_res
            )
        {}

};

#endif
#ifndef TEST_INNEROUTERSOLVE_H
#define TEST_INNEROUTERSOLVE_H

#include "test.h"

#include "solvers/IterativeSolve.h"
#include "solvers/nested/InnerOuterSolve.h"

using namespace cascade;

template <template <typename> typename TMatrix>
class InnerSolver_Mock: public GenericIterativeSolve<TMatrix>
{
public:

    Vector<double> incr_soln;

    InnerSolver_Mock(
        int iterations_to_converge,
        Vector<double> arg_soln,
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg

    ):
        incr_soln(
            arg_soln *
            Scalar<double>(
                static_cast<double>(1.) /
                static_cast<double>(iterations_to_converge)
            )
        ),
        GenericIterativeSolve<TMatrix>(arg_gen_lin_sys, arg_pkg)
    {}

    void derived_generic_reset() override {}

    void iterate() override {
        this->generic_soln += incr_soln;
    }

};

template <template <typename> typename TMatrix>
class InnerOuterSolve_Mock: public InnerOuterSolve<TMatrix>
{
public:

    bool init_inner_outer_hit = false;
    int outer_iterate_setup_hit_count = 0;
    int outer_iterate_complete_hit_count = 0;

    SolveArgPkg moving_args;

    Vector<double> soln_addition;
    int iterations_to_converge;

    using InnerOuterSolve<TMatrix>::iterate;

    void deal_with_nan_inner_solve() override {};

    void initialize_inner_outer_solver() override {
        init_inner_outer_hit = true;
        this->inner_solver = std::make_shared<InnerSolver_Mock<TMatrix>>(
            iterations_to_converge,
            soln_addition,
            this->gen_lin_sys_ptr,
            moving_args
        );
    }

    void outer_iterate_setup() override {
        outer_iterate_setup_hit_count += 1;
        this->inner_solver = std::make_shared<InnerSolver_Mock<TMatrix>>(
            iterations_to_converge,
            soln_addition,
            this->gen_lin_sys_ptr,
            moving_args
        );
    }

    void outer_iterate_complete() override {
        outer_iterate_complete_hit_count += 1;
        this->generic_soln = this->inner_solver->get_generic_soln();
        moving_args.init_guess = this->generic_soln;
    }

    InnerOuterSolve_Mock(
        Vector<double> arg_soln_addition,
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        soln_addition(arg_soln_addition),
        InnerOuterSolve<TMatrix>(arg_gen_lin_sys, arg_pkg),
        iterations_to_converge(arg_pkg.max_iter * arg_pkg.max_inner_iter),
        moving_args(
            arg_pkg.max_inner_iter,
            SolveArgPkg::default_max_inner_iter,
            arg_pkg.target_rel_res,
            arg_pkg.init_guess
        )
    {
        initialize_inner_outer_solver();
    }

};

#endif
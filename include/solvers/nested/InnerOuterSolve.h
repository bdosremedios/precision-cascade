#ifndef INNEROUTERSOLVE_H
#define INNEROUTERSOLVE_H

#include "../IterativeSolve.h"

namespace cascade {

template <template <typename> typename TMatrix>
class InnerOuterSolve: public GenericIterativeSolve<TMatrix>
{
protected:

    int max_inner_iter; // mutable to allow setting by derived solvers
    std::vector<std::vector<double>> inner_res_norm_hist;
    SolveArgPkg inner_solve_arg_pkg;
    std::shared_ptr<GenericIterativeSolve<TMatrix>> inner_solver;

    void iterate() override {
        outer_iterate_setup();
        inner_solver->solve();
        outer_iterate_complete();
    };

    // Initialize inner outer solver;
    virtual void initialize_inner_outer_solver() = 0;

    // Specify inner_solver for outer_iterate_calc and setup
    virtual void outer_iterate_setup() = 0;

    // Update generic_soln and inner_res_norm_hist according to derived
    virtual void outer_iterate_complete() = 0;

public:

    InnerOuterSolve(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg
    ): 
        max_inner_iter(arg_pkg.max_inner_iter),
        GenericIterativeSolve<TMatrix>(arg_gen_lin_sys, arg_pkg)
    {
        this->max_iter = (
            (arg_pkg.check_default_max_iter()) ?
            10 : arg_pkg.max_iter
        );
        // Create inner_solve_arg_pkg matching arg_pkg except with inner
        // iteration, set that as the inner solver's outer iteration
        inner_solve_arg_pkg = SolveArgPkg(
            max_inner_iter,
            SolveArgPkg::default_max_inner_iter,
            this->target_rel_res,
            this->init_guess
        );
    }

    // Forbid rvalue instantiation
    InnerOuterSolve(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&
    );

    std::vector<std::vector<double>> get_inner_res_norm_hist() const {
        return inner_res_norm_hist;
    };

};

}

#endif
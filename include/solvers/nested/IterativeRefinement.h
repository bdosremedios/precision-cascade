#ifndef ITERATIVEREFINEMENT_H
#define ITERATIVEREFINEMENT_H

#include "InnerOuterSolve.h"

namespace cascade {

template <template <typename> typename TMatrix>
class IterativeRefinement: public InnerOuterSolve<TMatrix>
{
protected:

    void outer_iterate_complete() override {

        // Add error back to generic_soln since that is solution of the
        // inner_solver under iterative refinement
        this->generic_soln += this->inner_solver->get_generic_soln();

        this->inner_res_norm_hist.push_back(
            this->inner_solver->get_res_norm_history()
        );

    }

    Vector<double> make_inner_IR_guess(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys
    ) const {
        // Matching iterative refinement algorithm should be zero guess
        return Vector<double>::Zero(
            arg_gen_lin_sys->get_cu_handles(),
            arg_gen_lin_sys->get_n()
        );
    }

public:

    IterativeRefinement(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        InnerOuterSolve<TMatrix>(arg_gen_lin_sys, arg_pkg)
    {
        // Replace initial guess with IR guess of zeroes for existing
        // inner_solve_arg_pkg
        this->inner_solve_arg_pkg.init_guess = make_inner_IR_guess(
            arg_gen_lin_sys
        );
    }

    // Forbid rvalue instantiation
    IterativeRefinement(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&
    );

};

}

#endif
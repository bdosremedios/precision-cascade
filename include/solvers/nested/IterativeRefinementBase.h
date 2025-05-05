#ifndef ITERATIVEREFINEMENTBASE_H
#define ITERATIVEREFINEMENTBASE_H

#include "InnerOuterSolve.h"

namespace cascade {

template <template <typename> typename TMatrix>
class IterativeRefinementBase: public InnerOuterSolve<TMatrix>
{
protected:

    virtual void outer_iterate_complete() override {

        // Add error back to generic_soln since that is solution of the
        // inner_solver under iterative refinement
        this->generic_soln += this->inner_solver->get_generic_soln();

    }

    virtual void deal_with_nan_inner_solve() override {

        // If an inner iteration failed by getting a nan results, simulate a
        // stagnated spin where no movement was made that took up the time of
        // iteration and do not update the solution
        std::vector<double> spin_vec = inner_solver->get_res_norm_history();
        for (int i=1; i<spin_vec.size(); i++) {
            spin_vec[i] = spin_vec[0];
        }
        inner_res_norm_history.push_back(spin_vec);
        inner_iterations.push_back(inner_solver->get_iteration());

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

    IterativeRefinementBase(
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
    IterativeRefinementBase(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&
    );

};

}

#endif
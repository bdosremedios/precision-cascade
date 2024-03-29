#ifndef ITERATIVEREFINEMENT_H
#define ITERATIVEREFINEMENT_H

#include "InnerOuterSolve.h"

template <template <typename> typename M>
class IterativeRefinement: public InnerOuterSolve<M>
{
protected:

    // *** Concrete Methods ***
    void outer_iterate_complete() override {

        // Add error back to generic_soln since that is solution of the inner_solver under
        // iterative refinement
        this->generic_soln += this->inner_solver->get_generic_soln();

        // Residual is b - inner_solver residual so update through that
        std::vector<double> temp;
        for (int i=0; i < this->inner_solver->get_iteration(); ++i) {
            temp.push_back(
                (this->lin_sys.get_b()-(this->inner_solver->get_res_hist()).get_col(i)).norm().get_scalar()
            );
        }
        this->inner_res_norm_hist.push_back(temp);

    }

    // Create initial guess for inner solver
    Vector<double> make_inner_IR_guess(GenericLinearSystem<M> const &arg_lin_sys) const {
        return Vector<double>::Zero(arg_lin_sys.get_handle(), arg_lin_sys.get_n());
    }

public:

    // *** Constructors ***
    IterativeRefinement(
        const GenericLinearSystem<M> &arg_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        InnerOuterSolve<M>(arg_lin_sys, arg_pkg)
    {
        // Replace initial guess with IR guess of zeroes for existing inner_solve_arg_pkg
        this->inner_solve_arg_pkg.init_guess = make_inner_IR_guess(arg_lin_sys);
    }

    // Forbid rvalue instantiation
    IterativeRefinement(const GenericLinearSystem<M> &&, const SolveArgPkg &);
    IterativeRefinement(const GenericLinearSystem<M> &, const SolveArgPkg &&);
    IterativeRefinement(const GenericLinearSystem<M> &&, const SolveArgPkg &&);

};

#endif
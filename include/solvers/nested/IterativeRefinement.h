#ifndef ITERATIVEREFINEMENT_H
#define ITERATIVEREFINEMENT_H

#include "InnerOuterSolve.h"

class IterativeRefinement: public InnerOuterSolve
{
protected:

    // *** PROTECTED OVERRIDE METHODS ***

    void outer_iterate_complete() override {

        // Add error back to generic_soln since that is solution of the inner_solver under
        // iterative refinement
        generic_soln += inner_solver->get_generic_soln();

        // Residual is b - inner_solver residual so update through that
        vector<double> temp;
        for (int i=0; i<inner_solver->get_iteration(); ++i) {
            temp.push_back(
                (b-(inner_solver->get_res_hist())(all, i)).norm()
            );
        }
        inner_res_norm_hist.push_back(temp);

    }

    // *** PROTECTED METHODS ***

    // Create initial guess for inner solver
    Matrix<double, Dynamic, 1> make_inner_IR_guess(Matrix<double, Dynamic, Dynamic> const &arg_A) const {
        return Matrix<double, Dynamic, 1>::Zero(arg_A.cols(), 1);
    }

public:

    // *** CONSTRUCTORS ***

    IterativeRefinement(
        Matrix<double, Dynamic, Dynamic> const &arg_A,
        Matrix<double, Dynamic, 1> const &arg_b,
        SolveArgPkg const &arg_pkg
    ):
        InnerOuterSolve(arg_A, arg_b, arg_pkg)
    {
        // Replace initial guess with IR for existing inner_solve_arg_pkg
        inner_solve_arg_pkg.init_guess = make_inner_IR_guess(A);
    }

};

#endif
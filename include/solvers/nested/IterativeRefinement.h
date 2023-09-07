#ifndef ITERATIVEREFINEMENT_H
#define ITERATIVEREFINEMENT_H

#include "InnerOuterSolve.h"

class IterativeRefinement: public InnerOuterSolve {

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
    
    public:

        // *** CONSTRUCTORS ***

        IterativeRefinement(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b, 
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            int const &arg_max_inner_iter,
            int const &arg_max_outer_iter,
            double const &arg_target_rel_res
        ):
            InnerOuterSolve(
                arg_A, arg_b, arg_init_guess,
                arg_max_inner_iter,
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}

};

#endif
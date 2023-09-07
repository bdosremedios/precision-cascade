#ifndef INNEROUTERSOLVE_H
#define INNEROUTERSOLVE_H

#include "IterativeSolve.h"

class InnerOuterSolve: public GenericIterativeSolve {

    protected:

        // *** PROTECTED ATTRIBUTES ***

        const int max_inner_iter;
        vector<vector<double>> inner_res_norm_hist;
        shared_ptr<GenericIterativeSolve> inner_solver;

        // *** PROTECTED OVERRIDE METHODS ***

        void iterate() override {

            outer_iterate_setup();
            inner_solver->solve();
            outer_iterate_complete();

        };

        // *** PROTECTED ABSTRACT METHODS ***

        // Initialize inner outer solver;
        virtual void initialize_inner_outer_solver() = 0;

        // Specify inner_solver for outer_iterate_calc and setup
        virtual void outer_iterate_setup() = 0;

        // Update generic_soln and inner_res_norm_hist according to derived
        virtual void outer_iterate_complete() = 0;

    public:

        // *** CONSTRUCTORS ***

        InnerOuterSolve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b, 
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            int const &arg_max_inner_iter=10,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ): 
            max_inner_iter(arg_max_inner_iter),
            GenericIterativeSolve(
                arg_A, arg_b, arg_init_guess,
                arg_max_outer_iter, arg_target_rel_res
            )
        {}

        // *** GETTERS ***
        vector<vector<double>> get_inner_res_norm_hist() const { return inner_res_norm_hist; };

};

#endif
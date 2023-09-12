#ifndef INNEROUTERSOLVE_H
#define INNEROUTERSOLVE_H

#include "../IterativeSolve.h"

class InnerOuterSolve: public GenericIterativeSolve {

    protected:

        // *** PROTECTED ATTRIBUTES ***

        int max_inner_iter; // mutable to allow setting by specific solvers
        vector<vector<double>> inner_res_norm_hist;
        SolveArgPkg inner_solve_arg_pkg;
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
            SolveArgPkg const &arg_pkg
        ): 
            max_inner_iter((arg_pkg.check_default_max_inner_iter()) ? 10 :
                                                                      arg_pkg.max_inner_iter),
            GenericIterativeSolve(arg_A, arg_b, arg_pkg)
        {
            max_iter = (arg_pkg.check_default_max_iter()) ? 10 : arg_pkg.max_iter;
            // Create inner_solve_arg_pkg matching arg_pkg except with set inner iteration
            // set as inner's outer iteration
            inner_solve_arg_pkg = SolveArgPkg(max_inner_iter,
                                              SolveArgPkg::default_max_inner_iter,
                                              target_rel_res,
                                              init_guess);
        }

        // *** GETTERS ***
        vector<vector<double>> get_inner_res_norm_hist() const { return inner_res_norm_hist; };

};

#endif
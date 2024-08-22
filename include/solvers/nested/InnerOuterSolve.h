#ifndef INNEROUTERSOLVE_H
#define INNEROUTERSOLVE_H

#include "../IterativeSolve.h"

#include <chrono>
#include <iostream>

namespace cascade {

template <template <typename> typename TMatrix>
class InnerOuterSolve: public GenericIterativeSolve<TMatrix>
{
protected:

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    int mk_1 = 0;
    int mk_2 = 0;
    int mk_3 = 0;
    int mk_4 = 0;

    void mark_start() {
        start = clock.now();
    }

    int mark_stop() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            clock.now() - start
        ).count();
    }

    int max_inner_iter; // mutable to allow setting by derived solvers
    std::vector<std::vector<double>> inner_res_norm_history;
    std::vector<int> inner_iterations;
    SolveArgPkg inner_solve_arg_pkg;
    std::shared_ptr<GenericIterativeSolve<TMatrix>> inner_solver;

    void iterate() override {
        mark_start();
        outer_iterate_setup();
        mk_1 += mark_stop(); 
        mark_start();
        inner_solver->solve();
        mk_2 += mark_stop(); 
        mark_start();
        inner_res_norm_history.push_back(inner_solver->get_res_norm_history());
        inner_iterations.push_back(inner_solver->get_iteration());
        mk_3 += mark_stop();
        mark_start();
        outer_iterate_complete();
        mk_4 += mark_stop();

        if (this->get_iteration() == this->max_iter) {
            int total = mk_1 + mk_2 + mk_3 + mk_4;
            std::cout << "Iter: " << this->get_iteration() << " | "
                      << "Mark 1: "
                      << static_cast<float>(mk_1)/static_cast<float>(total)
                      << " " << mk_1 << " | "
                      << "Mark 2: "
                      << static_cast<float>(mk_2)/static_cast<float>(total)
                      << " " << mk_2 << " | "
                      << "Mark 3: "
                      << static_cast<float>(mk_3)/static_cast<float>(total)
                      << " " << mk_3 << " | "
                      << "Mark 4: "
                      << static_cast<float>(mk_4)/static_cast<float>(total)
                      << " " << mk_4 << std::endl;
        }

    };

    void derived_generic_reset() override {
        inner_res_norm_history.clear();
        inner_iterations.clear();
    }

    // Initialize inner outer solver;
    virtual void initialize_inner_outer_solver() = 0;

    // Specify inner_solver for outer_iterate_calc and setup
    virtual void outer_iterate_setup() = 0;

    // Update generic_soln and inner_res_norm_history according to derived
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

    std::vector<std::vector<double>> get_inner_res_norm_history() const {
        return inner_res_norm_history;
    };

    std::vector<int> get_inner_iterations() const {
        return inner_iterations;
    };

};

}

#endif
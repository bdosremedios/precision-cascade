#ifndef FP_GMRES_IR_SOLVE_H
#define FP_GMRES_IR_SOLVE_H

#include "preconditioners/ImplementedPreconditioners.h"
#include "IterativeRefinement.h"
#include "GMRES.h"

#include <memory>

using std::shared_ptr, std::make_shared;

template <typename T, typename U=T>
class FP_GMRES_IR_Solve: public IterativeRefinement {

    private:

        // *** PRIVATE HELPER METHODS ***

        int determine_max_iter(int max_iter, Matrix<double, Dynamic, Dynamic> const &arg_A) const {
            if (max_iter == -1) {
                return arg_A.rows();
            } else {
                return max_iter;
            }
        }

        void set_inner_solve() {
            inner_solver = make_shared<GMRESSolve<T>>(
                A, res_hist(all, curr_iter), basis_zero_tol,
                left_precond_ptr, right_precond_ptr,
                max_inner_iter, target_rel_res
            );
        }

    protected:

        // *** PROTECTED ATTRIBUTES ***
        double basis_zero_tol;
        shared_ptr<Preconditioner<U>> left_precond_ptr;
        shared_ptr<Preconditioner<U>> right_precond_ptr;

        // *** PROTECTED OVERRIDE METHODS ***

        // Initialize inner outer solver;
        void initialize_inner_outer_solver() override { set_inner_solve(); }

        // Specify inner_solver for outer_iterate_calc and setup
        void outer_iterate_setup() override { set_inner_solve(); }

        void derived_generic_reset() override {} // Explicitly set as empty function
    
    public:

        // *** CONSTRUCTORS ***

        // Constructor without initial guess and no preconditioners
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            double const &arg_basis_zero_tol,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            FP_GMRES_IR_Solve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                determine_max_iter(arg_max_inner_iter, arg_A),
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and no preconditioners
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            double const &arg_basis_zero_tol,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            FP_GMRES_IR_Solve(
                arg_A, arg_b, arg_init_guess,
                arg_basis_zero_tol,
                make_shared<NoPreconditioner<T>>(),
                determine_max_iter(arg_max_inner_iter, arg_A),
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}
        
        // Constructor without initial guess and left preconditioner
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            double const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            FP_GMRES_IR_Solve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                arg_left_precond_ptr,
                determine_max_iter(arg_max_inner_iter, arg_A),
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and left preconditioner
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            double const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            FP_GMRES_IR_Solve(
                arg_A, arg_b, arg_init_guess,
                arg_basis_zero_tol,
                arg_left_precond_ptr, make_shared<NoPreconditioner<T>>(),
                determine_max_iter(arg_max_inner_iter, arg_A),
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}

        // Constructor without initial guess and both preconditioners
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            double const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            FP_GMRES_IR_Solve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                arg_left_precond_ptr, arg_right_precond_ptr,
                determine_max_iter(arg_max_inner_iter, arg_A),
                arg_max_outer_iter,
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and both preconditioners
        FP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            double const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            basis_zero_tol(arg_basis_zero_tol),
            left_precond_ptr(arg_left_precond_ptr),
            right_precond_ptr(arg_right_precond_ptr),
            IterativeRefinement(
                arg_A, arg_b, arg_init_guess,
                arg_max_inner_iter, arg_max_outer_iter, arg_target_rel_res
            )
        {}

};

#endif
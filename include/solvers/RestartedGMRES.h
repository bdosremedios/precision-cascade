#ifndef RESTARTEDGMRES_H
#define RESTARTEDGMRES_H

#include "Eigen/Dense"

#include "preconditioners/ImplementedPreconditioners.h"
#include "IterativeSolve.h"
#include "GMRES.h"

using Eigen::Matrix, Eigen::Dynamic;

template <typename T, typename U=T>
class FP_RSTRT_GMRESSolve: public TypedIterativeSolve<T> {

    protected:

        GMRES gmres;
    
    public:

        // *** CONSTRUCTORS ***

        // Constructor without initial guess and no preconditioners
        FP_RSTRT_GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_basis_zero_tol,
            int const &arg_max_inner_iter=-1,
            int const &arg_max_outer_iter=10,
            double const &arg_target_rel_res=1e-10
        ):
            max_outer_iter(arg_max_outer_iter)
        {
            curr_gmres = GMRESSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                arg_max_inner_iter,
                arg_target_rel_res
            )
        }

        // // Constructor with initial guess and no preconditioners
        // GMRESSolve(
        //     Matrix<T, Dynamic, Dynamic> const &arg_A,
        //     Matrix<T, Dynamic, 1> const &arg_b,
        //     Matrix<T, Dynamic, 1> const &arg_x_0,
        //     T const &arg_basis_zero_tol,
        //     int const &arg_max_outer_iter=-1,
        //     double const &arg_target_rel_res=1e-10
        // ):
        //     GMRESSolve(
        //         arg_A, arg_b, arg_x_0,
        //         arg_basis_zero_tol,
        //         make_shared<NoPreconditioner<T>>(),
        //         determine_max_iter(arg_max_outer_iter, arg_A),
        //         arg_target_rel_res
        //     )
        // {}
        
        // // Constructor without initial guess and left preconditioner
        // GMRESSolve(
        //     Matrix<T, Dynamic, Dynamic> const &arg_A,
        //     Matrix<T, Dynamic, 1> const &arg_b,
        //     T const &arg_basis_zero_tol,
        //     shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
        //     int const &arg_max_outer_iter=-1,
        //     double const &arg_target_rel_res=1e-10
        // ):
        //     GMRESSolve(
        //         arg_A, arg_b, this->make_guess(arg_A),
        //         arg_basis_zero_tol,
        //         arg_left_precond_ptr,
        //         determine_max_iter(arg_max_outer_iter, arg_A),
        //         arg_target_rel_res
        //     )
        // {}

        // // Constructor with initial guess and left preconditioner
        // GMRESSolve(
        //     Matrix<T, Dynamic, Dynamic> const &arg_A,
        //     Matrix<T, Dynamic, 1> const &arg_b,
        //     Matrix<T, Dynamic, 1> const &arg_x_0,
        //     T const &arg_basis_zero_tol,
        //     shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
        //     int const &arg_max_outer_iter=-1,
        //     double const &arg_target_rel_res=1e-10
        // ):
        //     GMRESSolve(
        //         arg_A, arg_b, arg_x_0,
        //         arg_basis_zero_tol,
        //         arg_left_precond_ptr, make_shared<NoPreconditioner<T>>(),
        //         determine_max_iter(arg_max_outer_iter, arg_A),
        //         arg_target_rel_res
        //     )
        // {}

        // // Constructor without initial guess and both preconditioners
        // GMRESSolve(
        //     Matrix<T, Dynamic, Dynamic> const &arg_A,
        //     Matrix<T, Dynamic, 1> const &arg_b,
        //     T const &arg_basis_zero_tol,
        //     shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
        //     shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
        //     int const &arg_max_outer_iter=-1,
        //     double const &arg_target_rel_res=1e-10
        // ):
        //     GMRESSolve(
        //         arg_A, arg_b, this->make_guess(arg_A),
        //         arg_basis_zero_tol,
        //         arg_left_precond_ptr, arg_right_precond_ptr,
        //         determine_max_iter(arg_max_outer_iter, arg_A),
        //         arg_target_rel_res
        //     )
        // {}

        // // Constructor with initial guess and both preconditioners
        // GMRESSolve(
        //     Matrix<T, Dynamic, Dynamic> const &arg_A,
        //     Matrix<T, Dynamic, 1> const &arg_b,
        //     Matrix<T, Dynamic, 1> const &arg_x_0,
        //     T const &arg_basis_zero_tol,
        //     shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
        //     shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
        //     int const &arg_max_outer_iter=-1,
        //     double const &arg_target_rel_res=1e-10
        // ):
        //     basis_zero_tol(arg_basis_zero_tol),
        //     left_precond_ptr(arg_left_precond_ptr),
        //     right_precond_ptr(arg_right_precond_ptr),
        //     TypedIterativeSolve<T>::TypedIterativeSolve(
        //         arg_A, arg_b, arg_x_0,
        //         determine_max_iter(arg_max_outer_iter, arg_A),
        //         arg_target_rel_res
        //     )
        // { initializeGMRES(); }

};

#endif
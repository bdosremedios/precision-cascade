#ifndef RESTARTEDGMRES_H
#define RESTARTEDGMRES_H

#include "Eigen/Dense"

#include "preconditioners/ImplementedPreconditioners.h"
#include "LinearSolve.h"
#include "GMRES.h"

using Eigen::Matrix, Eigen::Dynamic;

template <typename T, typename U=T>
class RestartedGMRES: public LinearSolve<T> {
    
    public:

        // Constructors
        RestartedGMRES(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr = make_shared<NoPreconditioner<T>>(),
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr = make_shared<NoPreconditioner<T>>()
        ):
            basis_zero_tol(arg_basis_zero_tol),
            left_precond_ptr(arg_left_precond_ptr),
            right_precond_ptr(arg_right_precond_ptr),
            LinearSolve<T>::LinearSolve(arg_A, arg_b)
        {
            constructorHelper();
        }

        RestartedGMRES(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr = make_shared<NoPreconditioner<T>>(),
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr = make_shared<NoPreconditioner<T>>()
        ):
            basis_zero_tol(arg_basis_zero_tol),
            left_precond_ptr(arg_left_precond_ptr),
            right_precond_ptr(arg_right_precond_ptr),
            LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0)
        {
            constructorHelper();
        }

};

#endif
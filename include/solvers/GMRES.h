#ifndef GMRES_H
#define GMRES_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>

using std::cout, std::endl;

template <typename T>
class GMRESSolve: public LinearSolve<T> {

    protected:
        Matrix<T, Dynamic, Dynamic> Q_kbasis;
        Matrix<T, Dynamic, Dynamic> H;
        int krylov_subspace_dim = 0;
        T basis_zero_tol;

        void update_subspace_and_convergence() {

            // Update krylov subspace
            if (krylov_subspace_dim == 0) {
                Matrix<T, Dynamic, 1> r_0(this->b - (this->A)*(this->x));
                // Check convergence and don't divide if converged
                if (r_0.norm() > basis_zero_tol) {
                    Q_kbasis(Eigen::placeholders::all, 0) = r_0/r_0.norm();
                    ++krylov_subspace_dim; // Update krylov dimension
                } else {
                    this->converged = true;
                }
            } else {
                int new_k = krylov_subspace_dim;
                Matrix<T, Dynamic, 1> q = (this->A)*Q_kbasis(Eigen::placeholders::all, new_k-1);
                for (int i=0; i<new_k; ++i) {
                    // MGS since newly orthogonalized q is used for next vector
                    H(i, new_k-1) = Q_kbasis(Eigen::placeholders::all, i).dot(q);
                    q -= H(i, new_k-1)*Q_kbasis(Eigen::placeholders::all, i);
                }
                H(new_k, new_k-1) = q.norm();

                // Check convergence and don't divide if converged
                if (H(new_k, new_k-1) > basis_zero_tol) {
                    Q_kbasis(Eigen::placeholders::all, new_k) = q/H(new_k, new_k-1);
                    ++krylov_subspace_dim; // Update krylov dimension
                } else {
                    this->converged = true;
                }

            }

        }

        void iterate() override {
            
            int new_k = krylov_subspace_dim;
            update_subspace_and_convergence();

        }
    
    public:
        // Constructors
        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b,
                   T arg_basis_zero_tol):
            basis_zero_tol(arg_basis_zero_tol), LinearSolve<T>::LinearSolve(arg_A, arg_b) {
            constructorHelper();
        }

        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b, 
                   const Matrix<T, Dynamic, 1> arg_x_0,
                   T arg_basis_zero_tol):
            basis_zero_tol(arg_basis_zero_tol), LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0) {
            constructorHelper();
        }

        void constructorHelper() {
            Q_kbasis = Matrix<T, Dynamic, Dynamic>::Zero(this->m, this->n);
            H = Matrix<T, Dynamic, Dynamic>::Zero(this->n+1, this->n);
        }

};

template <typename T>
class GMRESSolveTestingMock: public GMRESSolve<T> {

    public:
    // Constructors
    using GMRESSolve<T>::GMRESSolve;
    using GMRESSolve<T>::H;
    using GMRESSolve<T>::Q_kbasis;
    using GMRESSolve<T>::update_subspace_and_convergence;
    using GMRESSolve<T>::x;
    using GMRESSolve<T>::krylov_subspace_dim;

};

#endif
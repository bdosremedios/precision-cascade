#ifndef GMRES_H
#define GMRES_H

#include "LinearSolve.h"
#include "preconditioners/ImplementedPreconditioners.h"
#include "tools/Substitution.h"
#include "Eigen/Dense"

#include <iostream>
#include <memory>
#include <cmath>

using std::cout, std::endl;
using std::shared_ptr, std::make_shared;
using std::sqrt;

using Eigen::placeholders::all;

template <typename T, typename U=T>
class GMRESSolve: public LinearSolve<T> {

    protected:

        using LinearSolve<T>::m;
        using LinearSolve<T>::A;
        using LinearSolve<T>::b;
        using LinearSolve<T>::x;
        using LinearSolve<T>::x_0;

        shared_ptr<Preconditioner<U>> left_precond_ptr;
        shared_ptr<Preconditioner<U>> right_precond_ptr;

        Matrix<T, Dynamic, Dynamic> Q_kry_basis;
        Matrix<T, Dynamic, Dynamic> H;
        Matrix<T, Dynamic, Dynamic> Q_H;
        Matrix<T, Dynamic, Dynamic> R_H;
        Matrix<T, Dynamic, 1> next_q;

        int krylov_subspace_dim = 0;
        int max_krylov_subspace_dim;
        T basis_zero_tol;
        T rho;

        void update_subspace_k() {

            // Normalize next vector q and update subspace with it, assume that
            // checked in previous iteration that vector q was not zero vector
            // by checking H(k+1, k), with exception to zeroth iteration which
            // similarly checks the direct norm
            int k = krylov_subspace_dim-1;
            if (krylov_subspace_dim == 0) {
                Q_kry_basis(all, krylov_subspace_dim) = next_q/next_q.norm();
            } else {
                Q_kry_basis(all, krylov_subspace_dim) = next_q/H(k+1, k);
            }
            ++krylov_subspace_dim; // Update krylov dimension count
            
        }

        void update_nextq_and_Hkplus1() {

            int k = krylov_subspace_dim-1;
            
            // Find next vector power of linear system
            next_q = Q_kry_basis(all, k);
            next_q = right_precond_ptr->action_inv_M(next_q); // Apply action of right preconditioner
            next_q = A*next_q; // Apply matrix A
            next_q = left_precond_ptr->action_inv_M(next_q); // Apply action of left preconditioner

            // Orthogonlize next_q to previous basis vectors and store coefficients and
            // normalization in H for H_{kplus1, k} applying preconditioning
            for (int i=0; i<=k; ++i) {
                // MGS since newly orthogonalized q is used for orthogonalizing
                // each next vector
                H(i, k) = Q_kry_basis(all, i).dot(next_q);
                next_q -= H(i, k)*Q_kry_basis(all, i);
            }
            H(k+1, k) = next_q.norm();

        }

        void update_QR_fact() {

            // Initiate next column of QR fact as most recent of H
            int k = krylov_subspace_dim-1;
            R_H(all, k) = H(all, k);

            // Apply previous Given's rotations to new column
            // TODO: Can reduce run time by 2 since we know is upper triag
            R_H.block(0, k, k+1, 1) = Q_H.block(0, 0, k+1, k+1).transpose()*R_H.block(0, k, k+1, 1);

            // Apply the final Given's rotation manually making R_H upper triangular
            T alpha = R_H(k, k);
            T beta = R_H(k+1, k);
            T r_sqr = alpha*alpha + beta*beta; // Explicit intermediate variable to ensure
                                               // no auto casting into sqrt
            T r = sqrt(r_sqr);
            T c = alpha/r;
            T s = -beta/r;
            R_H(k, k) = r;
            R_H(k+1, k) = static_cast<T>(0);

            // Right multiply Q_H by transpose of new matrix to get updated Q_H
            // *will only modify last two columns so save time by just doing those 
            Matrix<T, 2, 2> givens_trans;
            givens_trans(0, 0) = c; givens_trans(0, 1) = s;
            givens_trans(1, 0) = -s; givens_trans(1, 1) = c;
            Q_H.block(0, k, k+2, 2) = Q_H.block(0, k, k+2, 2)*givens_trans;

        }

        void update_x_minimizing_res() {

            // Calculate RHS to solve
            int k = krylov_subspace_dim-1;
            Matrix<T, Dynamic, 1> rho_e1 = Matrix<T, Dynamic, 1>::Zero(k+2, 1);
            rho_e1(0) = rho;
            Matrix<T, Dynamic, 1> rhs = Q_H.block(0, 0, k+2, k+2).transpose()*rho_e1;

            // Use back substitution to solve
            Matrix<T, Dynamic, 1> y = back_substitution(R_H, rhs, krylov_subspace_dim);

            // Update x
            x = x_0 + Q_kry_basis.block(0, 0, m, krylov_subspace_dim)*y;

        }

        void check_termination() {

            // Check for termination condition with inability to expand subspace if
            // next basis vector is was in the existing Krylov subspace to basis_zero_tol
            int k = krylov_subspace_dim-1;
            if (H(k+1, k) <= basis_zero_tol) {
                this->terminated = true;
            }

        }

        void iterate() override {

            // Check isn't terminated and that solver isn't attempting to exceed
            // krylov subspace dimension, if is just do nothing
            if (!this->terminated) {
                if (krylov_subspace_dim < max_krylov_subspace_dim) {
                    update_subspace_k();
                    update_nextq_and_Hkplus1();
                    update_QR_fact();
                    update_x_minimizing_res();
                    check_termination();
                }
            }

        }
    
    public:

        // Constructors/Destructors
        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b,
                   T arg_basis_zero_tol):
            basis_zero_tol(arg_basis_zero_tol),
            LinearSolve<T>::LinearSolve(arg_A, arg_b),
            left_precond_ptr(make_shared<NoPreconditioner<T>>()),
            right_precond_ptr(make_shared<NoPreconditioner<T>>())
        {
            constructorHelper();
        }

        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b, 
                   const Matrix<T, Dynamic, 1> arg_x_0,
                   T arg_basis_zero_tol):
            basis_zero_tol(arg_basis_zero_tol),
            LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0),
            left_precond_ptr(make_shared<NoPreconditioner<T>>()),
            right_precond_ptr(make_shared<NoPreconditioner<T>>())
        {
            constructorHelper();
        }

        void constructorHelper() {

            // Pre-allocate all possible space needed to prevent memory
            // re-allocation
            Q_kry_basis = Matrix<T, Dynamic, Dynamic>::Zero(m, m);
            H = Matrix<T, Dynamic, Dynamic>::Zero(m+1, m);
            Q_H = Matrix<T, Dynamic, Dynamic>::Identity(m+1, m+1);
            R_H = Matrix<T, Dynamic, Dynamic>::Zero(m+1, m);

            // Specify max dimension for krylov subspace
            max_krylov_subspace_dim = m;

            // Set rho as initial residual norm
            Matrix<T, Dynamic, 1> r_0 = b - A*x_0;
            rho = r_0.norm();

            // Initialize next vector q as initial residual marking termination
            // since can not build Krylov subspace on zero vector
            next_q = r_0;
            if (next_q.norm() <= basis_zero_tol) {
                this->terminated = true;
            }

        }

};

template <typename T, typename U=T>
class GMRESSolveTestingMock: public GMRESSolve<T, U> {

    public:

        using GMRESSolve<T, U>::GMRESSolve;
        using GMRESSolve<T, U>::x;

        using GMRESSolve<T, U>::H;
        using GMRESSolve<T, U>::Q_kry_basis;
        using GMRESSolve<T, U>::Q_H;
        using GMRESSolve<T, U>::R_H;
        using GMRESSolve<T, U>::krylov_subspace_dim;
        using GMRESSolve<T, U>::max_krylov_subspace_dim;
        using GMRESSolve<T, U>::next_q;
        using GMRESSolve<T, U>::rho;
        using GMRESSolve<T, U>::iteration;

        using GMRESSolve<T, U>::update_QR_fact;
        using GMRESSolve<T, U>::update_x_minimizing_res;
        using GMRESSolve<T, U>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
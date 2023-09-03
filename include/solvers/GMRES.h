#ifndef GMRES_H
#define GMRES_H

#include "Eigen/Dense"

#include <iostream>
#include <memory>
#include <cmath>

#include "tools/Substitution.h"
#include "preconditioners/ImplementedPreconditioners.h"
#include "IterativeSolve.h"

using Eigen::Matrix, Eigen::Dynamic;
using std::cout, std::endl;
using std::shared_ptr, std::make_shared;
using std::sqrt;

using Eigen::placeholders::all;

template <typename T, typename U=T>
class GMRESSolve: public TypedIterativeSolve<T> {

    protected:

        using TypedIterativeSolve<T>::m;
        using TypedIterativeSolve<T>::A;
        using TypedIterativeSolve<T>::b;
        using TypedIterativeSolve<T>::typed_soln;
        using TypedIterativeSolve<T>::init_guess;
        using TypedIterativeSolve<T>::max_outer_iter;

        shared_ptr<Preconditioner<U>> left_precond_ptr;
        shared_ptr<Preconditioner<U>> right_precond_ptr;

        Matrix<T, Dynamic, Dynamic> Q_kry_basis;
        Matrix<T, Dynamic, Dynamic> H;
        Matrix<T, Dynamic, Dynamic> Q_H;
        Matrix<T, Dynamic, Dynamic> R_H;
        Matrix<T, Dynamic, 1> next_q;

        int kry_space_dim;
        int max_kry_space_dim;
        T basis_zero_tol;
        T rho;

        // *** PROTECTED INSTANTIATION HELPER FUNCTIONS ***

        int determine_max_iter(int max_iter, Matrix<T, Dynamic, Dynamic> const &arg_A) const {
            if (max_iter == -1) {
                return arg_A.rows();
            } else {
                return max_iter;
            }
        }

        void check_compatibility() const {

            // Assert compatibility of preconditioners with matrix
            if (!left_precond_ptr->check_compatibility_left(this->m)) {
                throw runtime_error("Left preconditioner is not compatible with linear system");
            }
            if (!right_precond_ptr->check_compatibility_right(this->n)) {
                throw runtime_error("Right preconditioner is not compatible with linear system");
            }
        
        }

        void set_initial_space() {

            // Set initial dim as 0
            kry_space_dim = 0;

            // Pre-allocate all possible space needed to prevent memory
            // re-allocation
            Q_kry_basis = Matrix<T, Dynamic, Dynamic>::Zero(m, m);
            H = Matrix<T, Dynamic, Dynamic>::Zero(m+1, m);
            Q_H = Matrix<T, Dynamic, Dynamic>::Identity(m+1, m+1);
            R_H = Matrix<T, Dynamic, Dynamic>::Zero(m+1, m);

            // Set rho as initial residual norm
            Matrix<T, Dynamic, 1> Minv_A_x0((left_precond_ptr->action_inv_M(A*init_guess)).template cast<T>());
            Matrix<T, Dynamic, 1> Minv_b((left_precond_ptr->action_inv_M(b)).template cast<T>());
            Matrix<T, Dynamic, 1> r_0(Minv_b - Minv_A_x0);
            rho = r_0.norm();

            // Initialize next vector q as initial residual marking termination
            // since can not build Krylov subspace on zero vector
            next_q = r_0;
            if (next_q.norm() <= basis_zero_tol) {
                this->terminated = true;
            }

        }

        void initializeGMRES() {

            // Specify max dimension for krylov subspace
            max_kry_space_dim = m;

            // Ensure that max_outer_iter does not exceed krylov subspace
            if (max_outer_iter > max_kry_space_dim) {
                throw runtime_error("GMRES outer iterations exceed matrix size");
            }

            check_compatibility();
            set_initial_space();

        }

        // *** PROTECTED ITERATION HELPER FUNCTIONS ***

        void update_subspace_k() {

            // Normalize next vector q and update subspace with it, assume that
            // checked in previous iteration that vector q was not zero vector
            // by checking H(k+1, k), with exception to zeroth iteration which
            // similarly checks the direct norm
            int k = kry_space_dim-1;
            if (kry_space_dim == 0) {
                Q_kry_basis(all, kry_space_dim) = next_q/next_q.norm();
            } else {
                Q_kry_basis(all, kry_space_dim) = next_q/H(k+1, k);
            }
            ++kry_space_dim; // Update krylov dimension count
            
        }

        void update_nextq_and_Hkplus1() {

            int k = kry_space_dim-1;
            
            // Find next vector power of linear system
            next_q = Q_kry_basis(all, k);
            next_q = (right_precond_ptr->action_inv_M(next_q)).template cast<T>(); // Apply action of right preconditioner
            next_q = A*next_q; // Apply matrix A
            next_q = (left_precond_ptr->action_inv_M(next_q)).template cast<T>(); // Apply action of left preconditioner

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
            int k = kry_space_dim-1;
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
            Matrix<T, Dynamic, 1> rho_e1 = Matrix<T, Dynamic, 1>::Zero(kry_space_dim+1, 1);
            rho_e1(0) = rho;
            Matrix<T, Dynamic, 1> rhs = (
                Q_H.block(0, 0, kry_space_dim+1, kry_space_dim+1).transpose()*rho_e1
            );

            // Use back substitution to solve
            Matrix<T, Dynamic, 1> y = back_substitution(
                static_cast<Matrix<T, Dynamic, Dynamic>>(R_H.block(0, 0, kry_space_dim, kry_space_dim)),
                // Trim off last entry of rhs for solve since solve of least squares problem
                // does not have coefficient to deal with it
                static_cast<Matrix<T, Dynamic, 1>>(rhs.block(0, 0, kry_space_dim, 1))
            );

            // Update typed_soln adjusting with right preconditioning
            typed_soln = (
                init_guess +
                (right_precond_ptr->action_inv_M(Q_kry_basis.block(0, 0, m, kry_space_dim)*y)).template cast<T>()
            );

        }

        // *** PROTECTED IMPLEMENTED OVERRIDING HELPER FUNCTIONS ***

        void check_termination() {

            // Check for termination condition with inability to expand subspace if
            // next basis vector is was in the existing Krylov subspace to basis_zero_tol
            int k = kry_space_dim-1;
            if (H(k+1, k) <= basis_zero_tol) {
                this->terminated = true;
            }

        }

        void iterate() override {

            // Check isn't terminated and that solver isn't attempting to exceed
            // krylov subspace dimension, if is just do nothing
            if (!this->terminated) {
                if (kry_space_dim < max_kry_space_dim) {
                    update_subspace_k();
                    update_nextq_and_Hkplus1();
                    update_QR_fact();
                    update_x_minimizing_res();
                    check_termination();
                }
            }

        }

        void derived_reset() override { set_initial_space(); } // Set derived reset to erase
                                                               // current krylov subspace made
    
    public:

        // *** CONSTRUCTORS ***

        // Constructor without initial guess and no preconditioners
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_basis_zero_tol,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            GMRESSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and no preconditioners
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            T const &arg_basis_zero_tol,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            GMRESSolve(
                arg_A, arg_b, arg_x_0,
                arg_basis_zero_tol,
                make_shared<NoPreconditioner<T>>(),
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        {}
        
        // Constructor without initial guess and left preconditioner
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            GMRESSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                arg_left_precond_ptr,
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and left preconditioner
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            GMRESSolve(
                arg_A, arg_b, arg_x_0,
                arg_basis_zero_tol,
                arg_left_precond_ptr, make_shared<NoPreconditioner<T>>(),
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        {}

        // Constructor without initial guess and both preconditioners
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            GMRESSolve(
                arg_A, arg_b, this->make_guess(arg_A),
                arg_basis_zero_tol,
                arg_left_precond_ptr, arg_right_precond_ptr,
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        {}

        // Constructor with initial guess and both preconditioners
        GMRESSolve(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            T const &arg_basis_zero_tol,
            shared_ptr<Preconditioner<U>> const &arg_left_precond_ptr,
            shared_ptr<Preconditioner<U>> const &arg_right_precond_ptr,
            int const &arg_max_outer_iter=-1,
            double const &arg_target_rel_res=1e-10
        ):
            basis_zero_tol(arg_basis_zero_tol),
            left_precond_ptr(arg_left_precond_ptr),
            right_precond_ptr(arg_right_precond_ptr),
            TypedIterativeSolve<T>::TypedIterativeSolve(
                arg_A, arg_b, arg_x_0,
                determine_max_iter(arg_max_outer_iter, arg_A),
                arg_target_rel_res
            )
        { initializeGMRES(); }

};

#endif
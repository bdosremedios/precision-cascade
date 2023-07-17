#ifndef GMRES_H
#define GMRES_H

#include "LinearSolve.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using std::cout, std::endl;
using Eigen::placeholders::all;
using std::sqrt, std::pow;

template <typename T>
class GMRESSolve: public LinearSolve<T> {

    protected:

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

            // Orthogonlize next_q to previous basis vectors and store coefficients and
            // normalization in H for H_{kplus1, k}
            int k = krylov_subspace_dim-1;
            next_q = (this->A)*Q_kry_basis(all, k);

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
            T a = R_H(k, k);
            T b = R_H(k+1, k);
            T r_sqr = a*a + b*b; // Explicit intermediate variable to ensure no auto casting into sqrt
            T r = sqrt(r_sqr);
            T c = a/r;
            T s = -b/r;
            R_H(k, k) = r;
            R_H(k+1, k) = 0;

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
            Matrix<T, Dynamic, 1> e1 = Matrix<T, Dynamic, 1>::Zero(k+2, 1);
            e1(0) = 1;
            Matrix<T, Dynamic, 1> rhs = rho*Q_H.block(0, 0, k+2, k+2).transpose()*e1;

            // Use back substitution to solve
            Matrix<T, Dynamic, 1> y = Matrix<T, Dynamic, 1>::Zero(krylov_subspace_dim, 1);
            for (int i=k; i>=0; --i) {
                y(i) = rhs(i);
                for (int j=i+1; j<=k; ++j) {
                    y(i) -= R_H(i, j)*y(j);
                }
                y(i) /= R_H(i, i);
            }

            // Update x
            this->x = this->x_0 + Q_kry_basis.block(0, 0, this->m, krylov_subspace_dim)*y;

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

            // Check matrix squareness
            if (this->m != this->n) { throw runtime_error("A not square"); };

            // Pre-allocate all possible space needed to prevent memory
            // re-allocation
            Q_kry_basis = Matrix<T, Dynamic, Dynamic>::Zero(this->m, this->n);
            H = Matrix<T, Dynamic, Dynamic>::Zero(this->n+1, this->n);
            Q_H = Matrix<T, Dynamic, Dynamic>::Identity(this->n+1, this->n+1);
            R_H = Matrix<T, Dynamic, Dynamic>::Zero(this->n+1, this->n);

            // Specify max dimension for krylov subspace
            max_krylov_subspace_dim = this->n;

            // Set rho as initial residual norm
            Matrix<T, Dynamic, 1> r_0 = this->b - (this->A)*(this->x_0);
            rho = r_0.norm();

            // Initialize next vector q as initial residual marking termination
            // since can not build Krylov subspace on zero vector
            next_q = r_0;
            if (next_q.norm() <= basis_zero_tol) {
                this->terminated = true;
            }

        }

};

template <typename T>
class GMRESSolveTestingMock: public GMRESSolve<T> {

    public:

        using GMRESSolve<T>::GMRESSolve;
        using GMRESSolve<T>::x;
        
        using GMRESSolve<T>::H;
        using GMRESSolve<T>::Q_kry_basis;
        using GMRESSolve<T>::Q_H;
        using GMRESSolve<T>::R_H;
        using GMRESSolve<T>::krylov_subspace_dim;
        using GMRESSolve<T>::max_krylov_subspace_dim;
        using GMRESSolve<T>::next_q;
        using GMRESSolve<T>::rho;
        using GMRESSolve<T>::iteration;

        using GMRESSolve<T>::update_QR_fact;
        using GMRESSolve<T>::update_x_minimizing_res;
        using GMRESSolve<T>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
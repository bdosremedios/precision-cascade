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
        T basis_zero_tol;
        T rho;

        void update_subspace_k() {

            // Protect against updating if converged already
            if (!this->converged) {

                // Update krylov subspace with orthonormalized new vector next_q
                Q_kry_basis(all, krylov_subspace_dim) = next_q;
                ++krylov_subspace_dim; // Update krylov dimension

            }
        
        }

        void update_next_q_Hkplus1_convergence() {

            // Protect against updating if converged already
            if (!this->converged) {

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

                // Check for convergence with exact being reached if next basis vector
                // is in the existing Krylov subspace, otherwise normalize next vector
                if (next_q.norm() > basis_zero_tol) {
                    next_q /= next_q.norm();
                } else {
                    this->converged = true;
                }

            }

        }

        void update_QR_fact() {

            // Initiate next column of QR fact as most recent of H
            int k = krylov_subspace_dim-1;
            R_H(all, k) = H(all, k);

            // Apply previous Given's rotations to new column
            // T prev_s = 0; // First Given's has no previous Given's so just modify by one
            // T prev_c = 1; // and set cos as initially zero so ondt add
            // Matrix<T, Dynamic, 1> test_R_H = R_H.col(k);
            // cout << "START LAST COL: " << endl;
            // cout << test_R_H << endl << endl;
            // cout << R_H.col(k) << endl << endl;
            // cout << Q_H.block(0, 0, k+1, k+1).transpose() << endl << endl;
            R_H.block(0, k, k+1, 1) = Q_H.block(0, 0, k+1, k+1).transpose()*R_H.block(0, k, k+1, 1);
            // for (int i=0; i<k; ++i) {

            //     // Get original entries on diagonal and subdiagonal from H corresponding to
            //     // ith Given's rotation
            //     T a_Givens = H(i, i);
            //     T b_Givens = H(i+1, i);

            //     // Modify diagonal value by previous Given's rotation second row since would be
            //     // effected by it
            //     a_Givens = a_Givens*prev_s+a_Givens*prev_c;

            //     // Calculate previous Given's rotation
            //     T r = sqrt(pow(a_Givens, 2) + pow(b_Givens, 2));
            //     T c = a_Givens/r;
            //     T s = -b_Givens/r;

            //     // Perform effect on new column
            //     T a_R = test_R_H(i);
            //     T b_R = test_R_H(i+1);
            //     // R_H(i, k) = a_R*c-b_R*s;
            //     // R_H(i+1, k) = a_R*s+b_R*c;
            //     test_R_H(i) = a_R*c-b_R*s;
            //     test_R_H(i+1) = a_R*s+b_R*c;

            //     // Store current Given's rotation as previous for next modification
            //     T prev_c = c;
            //     T prev_s = s;

            // }
            // I think this can be made to be O(n)
            
            // cout << "NEW LAST COL: " << endl;
            // cout << test_R_H << endl << endl;
            // cout << R_H.col(k) << endl << endl;

            // Apply the final Given's rotation manually making 
            // R_H upper triangular
            T a = R_H(k, k);
            T b = R_H(k+1, k);
            T r = sqrt(pow(a, 2) + pow(b, 2));
            T c = a/r;
            T s = -b/r;
            R_H(k, k) = r;
            R_H(k+1, k) = 0;

            // Right multiply Q_H by transpose of new matrix to get
            // updated Q_H
            Matrix<T, Dynamic, Dynamic> givens = Matrix<T, Dynamic, Dynamic>::Identity(k+2, k+2);
            givens(k, k) = c;
            givens(k, k+1) = -s;
            givens(k+1, k) = s;
            givens(k+1, k+1) = c;
            Q_H.block(0, 0, k+2, k+2) = Q_H.block(0, 0, k+2, k+2)*(givens.transpose());

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

        void iterate() override {
    
            update_subspace_k();
            update_next_q_Hkplus1_convergence();
            // Don't update if already converged
            if (!this->converged) {
                update_QR_fact();
                update_x_minimizing_res();
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

            // Pre-allocate all possible space needed to prevent memory
            // re-allocation
            Q_kry_basis = Matrix<T, Dynamic, Dynamic>::Zero(this->m, this->n);
            H = Matrix<T, Dynamic, Dynamic>::Zero(this->n+1, this->n);
            Q_H = Matrix<T, Dynamic, Dynamic>::Identity(this->n+1, this->n+1);
            R_H = Matrix<T, Dynamic, Dynamic>::Zero(this->n+1, this->n);

            // Set rho as initial residual norm
            Matrix<T, Dynamic, 1> r_0 = this->b - (this->A)*(this->x_0);
            rho = r_0.norm();

            // Initialize next vector q as initial residual marking convergence
            // if residual is already zero
            next_q = r_0;
            if (next_q.norm() > basis_zero_tol) {
                next_q = next_q/next_q.norm();
            } else {
                this->converged = true;
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
        using GMRESSolve<T>::next_q;
        using GMRESSolve<T>::rho;
    
        using GMRESSolve<T>::update_subspace_k;
        using GMRESSolve<T>::update_next_q_Hkplus1_convergence;
        using GMRESSolve<T>::update_QR_fact;

};

#endif
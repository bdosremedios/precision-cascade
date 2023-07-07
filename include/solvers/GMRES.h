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

            // Apply prev Given's rotations to new column
            // for (int i=0; i<k; ++i) {
            //     T a = H(i, i);
            //     T b = H(i+1, i);
            //     T r = sqrt(pow(a, 2) + pow(b, 2));
            //     T c = a/r;
            //     T s = -b/r;
            //     a = R_H(i, k);
            //     b = R_H(i+1, k);
            //     R_H(i, k) = a*c-b*s;
            //     R_H(i+1, k) = a*s+b*c;
            // }
            R_H.block(0, k, k+2, 1) = Q_H.block(0, 0, k+2, k+2).transpose()*R_H.block(0, k, k+2, 1);

            // Apply the final Given's rotation manually making 
            // R_H upper triangular
            T a = R_H(k, k);
            T b = R_H(k+1, k);
            T r = sqrt(pow(a, 2) + pow(b, 2));
            T c = a/r;
            T s = -b/r;
            // Matrix<T, Dynamic, Dynamic> givens_ = Matrix<T, Dynamic, Dynamic>::Identity(k+2, k+2);
            // givens_(k, k) = c;
            // givens_(k, k+1) = -s;
            // givens_(k+1, k) = s;
            // givens_(k+1, k+1) = c;
            // R_H.block(0, k, k+2, 1) = givens_*R_H.block(0, k, k+2, 1);
            R_H(k, k) = r;
            R_H(k+1, k) = 0;

            // Right multiply Q_H by transpose of new matrix to get
            // updated Q_H
            Matrix<T, Dynamic, Dynamic> givens = Matrix<T, Dynamic, Dynamic>::Identity(k+2, k+2);
            givens(k, k) = c;
            givens(k, k+1) = -s;
            givens(k+1, k) = s;
            givens(k+1, k+1) = c;
            // cout << Q_H.block(0, 0, k+2, k+2) << endl;
            Q_H.block(0, 0, k+2, k+2) = Q_H.block(0, 0, k+2, k+2)*(givens.transpose());
            // cout << Q_H.block(0, 0, k+2, k+2)*givens << endl;
            // Q_H = Q_H*(givens.transpose());

        }

        void calc_x_minimizing_res() {}

        void iterate() override {
    
            update_subspace_k();
            update_next_q_Hkplus1_convergence();
            // Don't update if already converged
            if (!this->converged) {
                update_QR_fact();
                calc_x_minimizing_res();
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

            // Initialize next vector q as initial residual marking convergence
            // if residual is already zero
            next_q = this->b - (this->A)*(this->x_0);
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
    
        using GMRESSolve<T>::update_subspace_k;
        using GMRESSolve<T>::update_next_q_Hkplus1_convergence;
        using GMRESSolve<T>::update_QR_fact;

};

#endif
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

        void update_krylov_subspace() {

            // Update krylov subspace
            if (krylov_subspace_dim == 0) {
                Matrix<T, Dynamic, 1> r_0(this->b - (this->A)*(this->x));
                Q_kbasis(Eigen::placeholders::all, 0) = r_0/r_0.norm();
            } else {
                int new_k = krylov_subspace_dim;
                Matrix<T, Dynamic, 1> q = (this->A)*Q_kbasis(Eigen::placeholders::all, new_k-1);
                for (int i=0; i<new_k; ++i) {
                    // MGS since newly orthogonalized q is used for next vector
                    H(i, new_k-1) = Q_kbasis(Eigen::placeholders::all, i).dot(q);
                    q -= H(i, new_k-1)*Q_kbasis(Eigen::placeholders::all, i);
                }
                H(new_k, new_k-1) = q.norm();
                Q_kbasis(Eigen::placeholders::all, new_k) = q/H(new_k, new_k-1);
            }

            // Update krylov dimension
            ++krylov_subspace_dim;

        }

        void iterate() override {

            update_krylov_subspace();

        }
    
    public:
        // Constructors
        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b): LinearSolve<T>::LinearSolve(arg_A, arg_b) {
            constructorHelper();
        }

        GMRESSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                   const Matrix<T, Dynamic, 1> arg_b, 
                   const Matrix<T, Dynamic, 1> arg_x_0): LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0) {
            constructorHelper();
        }

        void constructorHelper() {
            Q_kbasis = Matrix<T, Dynamic, Dynamic>(this->m, this->n);
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
    using GMRESSolve<T>::update_krylov_subspace;

};

#endif
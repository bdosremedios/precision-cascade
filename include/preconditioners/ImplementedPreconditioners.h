#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Preconditioner.h"

#include <cmath>

using std::abs;

template <typename U>
class NoPreconditioner: public Preconditioner<U> {

    public:

        using Preconditioner<U>::Preconditioner;
    
        Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const override {
            return vec;
        }

        bool check_compatibility_left(int m) const override { return true; };
        bool check_compatibility_right(int n) const override { return true; };

};

template <typename U>
class MatrixInverse: public Preconditioner<U> {

    public:

        Matrix<U, Dynamic, Dynamic> inv_M;

        MatrixInverse(Matrix<U, Dynamic, Dynamic> arg_inv_M): inv_M(arg_inv_M) {}

        bool check_compatibility_left(int arg_m) const override {
            return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
        };

        bool check_compatibility_right(int arg_n) const override {
            return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
        };

        Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const override {
            return inv_M*vec;
        }

};

template <typename T>
class ILU {

    public:

        Matrix<T, Dynamic, Dynamic> LU;

        ILU(Matrix<T, Dynamic, Dynamic> A, T drop_tol, T zero_tol) {
            
            int m = A.rows(); 
            LU = A;

            // Use IKJ variant for better predictability of modification
            for (int i=1; i<m; ++i) {
                for (int k=0; k<=i-1; ++k) {
                    if (abs(LU(k, k)) > zero_tol) {
                        T coeff = LU(i, k)/LU(k, k);
                        LU(i, k) = coeff;
                        for (int j=k+1; j<m; ++j) {
                            if (abs(A(i, j)) > zero_tol) {
                                LU(i, j) = LU(i, j) - coeff*LU(k, j);
                            }
                        }
                    } else {
                        throw runtime_error("ILU encountered zero diagonal entry");
                    }
                }
            }

        }

        Matrix<T, Dynamic, Dynamic> get_L() {

            int m = LU.rows(); 
            Matrix<T, Dynamic, Dynamic> L(Matrix<T, Dynamic, Dynamic>::Identity(m, m));
            for (int i=0; i<m; ++i) {
                for (int j=0; j<i; ++j) {
                    L(i, j) = LU(i, j);
                }
            }

            return L;

        }

        Matrix<T, Dynamic, Dynamic> get_U() {

            int m = LU.rows(); 
            Matrix<T, Dynamic, Dynamic> U(Matrix<T, Dynamic, Dynamic>::Zero(m, m));
            for (int i=0; i<m; ++i) {
                for (int j=i; j<m; ++j) {
                    U(i, j) = LU(i, j);
                }
            }

            return U;

        }

        virtual Matrix<T, Dynamic, 1> action_inv_M(Matrix<T, Dynamic, 1> vec) const {
            return Matrix<T, Dynamic, 1>::Zero(1, 1);
        }

        virtual bool check_compatibility_left(int m) const { return false; };
        virtual bool check_compatibility_right(int n) const { return false; };

};

#endif
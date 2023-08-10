#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Preconditioner.h"
#include "tools/Substitution.h"

#include <cmath>

using std::abs;

template <typename T>
class NoPreconditioner: public Preconditioner<T> {

    public:

        using Preconditioner<T>::Preconditioner;
    
        Matrix<T, Dynamic, 1> action_inv_M(Matrix<T, Dynamic, 1> const &vec) const override {
            return vec;
        }

        bool check_compatibility_left(int const &arg_m) const override { return true; };
        bool check_compatibility_right(int const &arg_n) const override { return true; };

};

template <typename T>
class MatrixInverse: public Preconditioner<T> {

    public:

        Matrix<T, Dynamic, Dynamic> inv_M;

        MatrixInverse(Matrix<T, Dynamic, Dynamic> const &arg_inv_M): inv_M(arg_inv_M) {}

        bool check_compatibility_left(int const &arg_m) const override {
            return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
        };

        bool check_compatibility_right(int const &arg_n) const override {
            return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
        };

        Matrix<T, Dynamic, 1> action_inv_M(Matrix<T, Dynamic, 1> const &vec) const override {
            return inv_M*vec;
        }

};

template <typename T>
class ILU: public Preconditioner<T> {

    public:

        Matrix<T, Dynamic, Dynamic> LU;
        int m;

        ILU(Matrix<T, Dynamic, Dynamic> const &A, T const &drop_tol, T const &zero_tol) {
            
            if (A.rows() != A.cols()) {
                throw runtime_error("Non square matrix A");
            }

            m = A.rows(); 
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

            Matrix<T, Dynamic, Dynamic> L(Matrix<T, Dynamic, Dynamic>::Identity(m, m));
            for (int i=0; i<m; ++i) {
                for (int j=0; j<i; ++j) {
                    L(i, j) = LU(i, j);
                }
            }

            return L;

        }

        Matrix<T, Dynamic, Dynamic> get_U() {

            Matrix<T, Dynamic, Dynamic> U(Matrix<T, Dynamic, Dynamic>::Zero(m, m));
            for (int i=0; i<m; ++i) {
                for (int j=i; j<m; ++j) {
                    U(i, j) = LU(i, j);
                }
            }

            return U;

        }

        Matrix<T, Dynamic, 1> action_inv_M(Matrix<T, Dynamic, 1> const &vec) const override {
            return Matrix<T, Dynamic, 1>::Zero(1, 1);
        }

        bool check_compatibility_left(int const &arg_m) const override {
            return arg_m == m;
        };

        bool check_compatibility_right(int const &arg_n) const override {
            return arg_n == m;
        };

};

#endif
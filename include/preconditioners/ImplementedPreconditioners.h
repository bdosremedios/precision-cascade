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

        Matrix<T, Dynamic, Dynamic> L;
        Matrix<T, Dynamic, Dynamic> U;
        int m;

        ILU(Matrix<T, Dynamic, Dynamic> const &A, T const &drop_tol, T const &zero_tol) {
            
            if (A.rows() != A.cols()) { throw runtime_error("Non square matrix A"); }

            m = A.rows();
            L = Matrix<T, Dynamic, Dynamic>::Identity(m, m);
            U = Matrix<T, Dynamic, Dynamic>::Zero(m, m);

            // Set U to be A and perform ILU
            U = A;

            // Use IKJ variant for better predictability of modification
            for (int i=1; i<m; ++i) {
                for (int k=0; k<=i-1; ++k) {
                    if (abs(U(k, k)) > zero_tol) {
                        T coeff = U(i, k)/U(k, k);
                        U(i, k) = 0;
                        L(i, k) = coeff;
                        for (int j=k+1; j<m; ++j) {
                            if (abs(U(i, j)) > zero_tol) {
                                U(i, j) = U(i, j) - coeff*U(k, j);
                            }
                        }
                    } else {
                        throw runtime_error("ILU encountered zero diagonal entry");
                    }
                }
            }

        }

        Matrix<T, Dynamic, 1> action_inv_M(Matrix<T, Dynamic, 1> const &vec) const override {
            return frwd_substitution(L, back_substitution(U, vec));
        }

        Matrix<T, Dynamic, Dynamic> get_L() const { return L; }
        Matrix<T, Dynamic, Dynamic> get_U() const { return U; }

        bool check_compatibility_left(int const &arg_m) const override { return arg_m == m; };
        bool check_compatibility_right(int const &arg_n) const override { return arg_n == m; };

};

#endif
#ifndef IMPLEMENTED_PRECONDITIONERS_H
#define IMPLEMENTED_PRECONDITIONERS_H

#include "Preconditioner.h"

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

#endif
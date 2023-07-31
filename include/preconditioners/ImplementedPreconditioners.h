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

        bool check_compatibility_left(int m, int n) const override { return true; };
        bool check_compatibility_right(int m, int n) const override { return true; };

};

// template <typename U>
// class MatrixInverse: public Preconditioner<U> {

//     public:

//         MatrixInverse(Matrix<U, Dynamic, Dynamic> arg_inv_M) {

//         }

//         Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const override {
//             return vec;
//         }

// };

#endif
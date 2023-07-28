#ifndef NOPRECONDITIONER_H
#define NOPRECONDITIONER_H

#include "Preconditioner.h"

template <typename U>
class NoPreconditioner: public LeftPreconditioner<U> {

    public:

        using LeftPreconditioner<U>::LeftPreconditioner;
    
        Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const override {
            return vec;
        }

};

#endif
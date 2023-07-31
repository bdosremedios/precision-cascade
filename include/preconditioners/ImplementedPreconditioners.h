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

};

#endif
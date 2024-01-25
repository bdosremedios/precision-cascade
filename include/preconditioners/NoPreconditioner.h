#ifndef NO_PRECONDITIONER_H
#define NO_PRECONDITIONER_H

#include "Preconditioner.h"

template <template <typename> typename M, typename W>
class NoPreconditioner: public Preconditioner<M, W>
{
public:

    using Preconditioner<M, W>::Preconditioner;

    Vector<W> action_inv_M(Vector<W> const &vec) const override {
        return vec;
    }

    bool check_compatibility_left(int const &arg_m) const override { return true; };
    bool check_compatibility_right(int const &arg_n) const override { return true; };

};

#endif
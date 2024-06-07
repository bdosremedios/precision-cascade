#ifndef NO_PRECONDITIONER_H
#define NO_PRECONDITIONER_H

#include "Preconditioner.h"

template <template <typename> typename TMatrix, typename TPrecision>
class NoPreconditioner:
    public Preconditioner<TMatrix, TPrecision>
{
public:

    using Preconditioner<TMatrix, TPrecision>::Preconditioner;

    Vector<TPrecision> action_inv_M(
        Vector<TPrecision> const &vec
    ) const override {
        return vec;
    }

    bool check_compatibility_left(int const &arg_m) const override {
        return true;
    };
    bool check_compatibility_right(int const &arg_n) const override {
        return true;
    };

    NoPreconditioner<TMatrix, double> * cast_dbl_ptr() const override {
        return new NoPreconditioner<TMatrix, double>();
    }

    NoPreconditioner<TMatrix, float> * cast_sgl_ptr() const override {
        return new NoPreconditioner<TMatrix, float>();
    }

    NoPreconditioner<TMatrix, __half> * cast_hlf_ptr() const override {
        return new NoPreconditioner<TMatrix, __half>();
    }

};

#endif
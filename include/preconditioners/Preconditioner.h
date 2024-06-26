#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "../types/types.h"

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class Preconditioner
{
public:

    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    virtual Vector<TPrecision> action_inv_M(Vector<TPrecision> const &vec) const = 0;
    virtual bool check_compatibility_left(int const &m) const = 0;
    virtual bool check_compatibility_right(int const &n) const = 0;

    virtual Preconditioner<TMatrix, double> * cast_dbl_ptr() const = 0;
    virtual Preconditioner<TMatrix, float> * cast_sgl_ptr() const = 0;
    virtual Preconditioner<TMatrix, __half> * cast_hlf_ptr() const = 0;

};

}

#endif
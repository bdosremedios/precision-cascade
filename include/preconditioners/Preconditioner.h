#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <memory>

#include "../types/types.h"

template <template <typename> typename M, typename W>
class Preconditioner
{
public:

    // *** Constructors ***

    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    // *** Virtual Abstract Methods *** 

    virtual Vector<W> action_inv_M(Vector<W> const &vec) const = 0;
    virtual bool check_compatibility_left(int const &m) const = 0;
    virtual bool check_compatibility_right(int const &n) const = 0;

    virtual Preconditioner<M, double> * cast_dbl_ptr() const = 0;
    virtual Preconditioner<M, float> * cast_sgl_ptr() const = 0;
    virtual Preconditioner<M, __half> * cast_hlf_ptr() const = 0;

};

#endif
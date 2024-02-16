#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "../types/types.h"

template <template <typename> typename M, typename W>
class Preconditioner
{
public:

    // *** Constructors ***
    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    // *** Virtual abstract methods *** 
    virtual Vector<W> action_inv_M(Vector<W> const &vec) const = 0;
    virtual bool check_compatibility_left(int const &m) const = 0;
    virtual bool check_compatibility_right(int const &n) const = 0;

    template <typename T>
    Vector<T> casted_action_inv_M(Vector<W> const &vec) { return action_inv_M(vec).template cast<T>(); }


};

#endif
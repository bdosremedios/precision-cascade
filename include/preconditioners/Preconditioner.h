#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "../types/types.h"

template <template <typename> typename M, typename W>
class Preconditioner
{
public:

    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    // Abstract function to apply action of inverse M on given vector
    virtual MatrixVector<W> action_inv_M(MatrixVector<W> const &vec) const = 0;

    // Abstract functions to check compatibility of preconditioner with linear system
    // on both left and right
    virtual bool check_compatibility_left(int const &m) const = 0;
    virtual bool check_compatibility_right(int const &n) const = 0;

};

#endif
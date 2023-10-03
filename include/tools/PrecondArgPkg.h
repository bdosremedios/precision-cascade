#ifndef PRECONDARGPKG_H
#define PRECONDARGPKG_H

#include "Eigen/Dense"

#include "../types/types.h"
#include "../preconditioners/ImplementedPreconditioners.h"

#include <memory>

using std::make_shared, std::shared_ptr;

template <template <typename> typename M, typename U>
class PrecondArgPkg
{
public:

    shared_ptr<Preconditioner<M, U>> left_precond;
    shared_ptr<Preconditioner<M, U>> right_precond;

    // *** CONSTRUCTORS ***

    PrecondArgPkg(
        shared_ptr<Preconditioner<M, U>> arg_left_precond = make_shared<NoPreconditioner<M, U>>(),
        shared_ptr<Preconditioner<M, U>> arg_right_precond = make_shared<NoPreconditioner<M, U>>()
    ):
        left_precond(arg_left_precond),
        right_precond(arg_right_precond)
    {};

};

#endif
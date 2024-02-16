#ifndef PRECONDARGPKG_H
#define PRECONDARGPKG_H

#include "../../types/types.h"
#include "../../preconditioners/implemented_preconditioners.h"

#include <memory>

template <template <typename> typename M, typename U>
class PrecondArgPkg
{
public:

    std::shared_ptr<Preconditioner<M, U>> left_precond;
    std::shared_ptr<Preconditioner<M, U>> right_precond;

    // *** Constructors ***
    PrecondArgPkg(
        std::shared_ptr<Preconditioner<M, U>> arg_left_precond = std::make_shared<NoPreconditioner<M, U>>(),
        std::shared_ptr<Preconditioner<M, U>> arg_right_precond = std::make_shared<NoPreconditioner<M, U>>()
    ):
        left_precond(arg_left_precond),
        right_precond(arg_right_precond)
    {};

};

#endif
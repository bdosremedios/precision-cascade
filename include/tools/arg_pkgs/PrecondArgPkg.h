#ifndef PRECONDARGPKG_H
#define PRECONDARGPKG_H

#include "../../types/types.h"
#include "../../preconditioners/implemented_preconditioners.h"

#include <cuda_fp16.h>

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

    ~PrecondArgPkg() {
        left_precond.reset();
        right_precond.reset();
    }

    PrecondArgPkg<M, double> * cast_dbl_ptr() const {
        return new PrecondArgPkg<M, double>(
            std::shared_ptr<Preconditioner<M, double>>(left_precond->cast_dbl_ptr()),
            std::shared_ptr<Preconditioner<M, double>>(right_precond->cast_dbl_ptr())
        );
    }

    PrecondArgPkg<M, float> * cast_sgl_ptr() const {
        return new PrecondArgPkg<M, float>(
            std::shared_ptr<Preconditioner<M, float>>(left_precond->cast_sgl_ptr()),
            std::shared_ptr<Preconditioner<M, float>>(right_precond->cast_sgl_ptr())
        );
    }

    PrecondArgPkg<M, __half> * cast_hlf_ptr() const {
        return new PrecondArgPkg<M, __half>(
            std::shared_ptr<Preconditioner<M, __half>>(left_precond->cast_hlf_ptr()),
            std::shared_ptr<Preconditioner<M, __half>>(right_precond->cast_hlf_ptr())
        );
    }

};

#endif
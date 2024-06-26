#ifndef PRECONDARGPKG_H
#define PRECONDARGPKG_H

#include "../../types/types.h"
#include "../../preconditioners/implemented_preconditioners.h"

#include <cuda_fp16.h>

#include <memory>

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class PrecondArgPkg
{
public:

    std::shared_ptr<Preconditioner<TMatrix, TPrecision>> left_precond;
    std::shared_ptr<Preconditioner<TMatrix, TPrecision>> right_precond;

    PrecondArgPkg(
        std::shared_ptr<Preconditioner<TMatrix, TPrecision>> arg_left_precond = (
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>()
        ),
        std::shared_ptr<Preconditioner<TMatrix, TPrecision>> arg_right_precond = (
            std::make_shared<NoPreconditioner<TMatrix, TPrecision>>()
        )
    ):
        left_precond(arg_left_precond),
        right_precond(arg_right_precond)
    {};

    ~PrecondArgPkg() {
        left_precond.reset();
        right_precond.reset();
    }

    PrecondArgPkg<TMatrix, double> * cast_dbl_ptr() const {
        return new PrecondArgPkg<TMatrix, double>(
            std::shared_ptr<Preconditioner<TMatrix, double>>(
                left_precond->cast_dbl_ptr()
            ),
            std::shared_ptr<Preconditioner<TMatrix, double>>(
                right_precond->cast_dbl_ptr()
            )
        );
    }

    PrecondArgPkg<TMatrix, float> * cast_sgl_ptr() const {
        return new PrecondArgPkg<TMatrix, float>(
            std::shared_ptr<Preconditioner<TMatrix, float>>(
                left_precond->cast_sgl_ptr()
            ),
            std::shared_ptr<Preconditioner<TMatrix, float>>(
                right_precond->cast_sgl_ptr()
            )
        );
    }

    PrecondArgPkg<TMatrix, __half> * cast_hlf_ptr() const {
        return new PrecondArgPkg<TMatrix, __half>(
            std::shared_ptr<Preconditioner<TMatrix, __half>>(
                left_precond->cast_hlf_ptr()
            ),
            std::shared_ptr<Preconditioner<TMatrix, __half>>(
                right_precond->cast_hlf_ptr()
            )
        );
    }

};

}

#endif
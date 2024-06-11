#ifndef MATRIX_INVERSE_PRECONDITIONER_H
#define MATRIX_INVERSE_PRECONDITIONER_H

#include "Preconditioner.h"

template <template <typename> typename TMatrix, typename TPrecision>
class MatrixInversePreconditioner:
    public Preconditioner<TMatrix, TPrecision>
{
protected:

    TMatrix<TPrecision> inv_M;

public:

    MatrixInversePreconditioner(
        TMatrix<TPrecision> const &arg_inv_M
    ): inv_M(arg_inv_M) {}

    Vector<TPrecision> action_inv_M(
        Vector<TPrecision> const &vec
    ) const override {
        return inv_M*vec;
    }

    bool check_compatibility_left(int const &arg_m) const override {
        return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
    };

    bool check_compatibility_right(int const &arg_n) const override {
        return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
    };

    MatrixInversePreconditioner<TMatrix, double> * cast_dbl_ptr() const override {
        return new MatrixInversePreconditioner<TMatrix, double>(
            inv_M.template cast<double>()
        );
    }

    MatrixInversePreconditioner<TMatrix, float> * cast_sgl_ptr() const override {
        return new MatrixInversePreconditioner<TMatrix, float>(
            inv_M.template cast<float>()
        );
    }

    MatrixInversePreconditioner<TMatrix, __half> * cast_hlf_ptr() const override {
        return new MatrixInversePreconditioner<TMatrix, __half>(
            inv_M.template cast<__half>()
        );
    }

};

#endif
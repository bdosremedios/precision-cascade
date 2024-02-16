#ifndef MATRIX_INVERSE_PRECONDITIONER_H
#define MATRIX_INVERSE_PRECONDITIONER_H

#include "Preconditioner.h"

template <template <typename> typename M, typename W>
class MatrixInversePreconditioner: public Preconditioner<M, W>
{
public:

    M<W> inv_M;

    // *** Constructors ***
    MatrixInversePreconditioner(M<W> const &arg_inv_M): inv_M(arg_inv_M) {}

    bool check_compatibility_left(int const &arg_m) const override {
        return ((inv_M.cols() == arg_m) && (inv_M.rows() == arg_m));
    };

    bool check_compatibility_right(int const &arg_n) const override {
        return ((inv_M.cols() == arg_n) && (inv_M.rows() == arg_n));
    };

    Vector<W> action_inv_M(Vector<W> const &vec) const override {
        return inv_M*vec;
    }

};

#endif
#ifndef ILU_PRECONDITIONER_H
#define ILU_PRECONDITIONER_H

#include "Preconditioner.h"
#include "tools/ILU_subroutines.h"

template <template <typename> typename TMatrix, typename TPrecision>
class ILUPreconditioner: public Preconditioner<TMatrix, TPrecision>
{
private:

    int m;
    TMatrix<TPrecision> L = TMatrix<TPrecision>(cuHandleBundle());
    TMatrix<TPrecision> U = TMatrix<TPrecision>(cuHandleBundle());
    TMatrix<TPrecision> P = TMatrix<TPrecision>(cuHandleBundle());

public:

    // ILU constructor taking premade L and U and no permutation
    ILUPreconditioner(
        const TMatrix<TPrecision> &arg_L,
        const TMatrix<TPrecision> &arg_U
    ):
        ILUPreconditioner(
            arg_L,
            arg_U,
            TMatrix<TPrecision>::Identity(
                arg_L.get_cu_handles(), arg_L.rows(), arg_L.rows()
            )
        )
    {}

    // ILU constructor taking premade L, U, and P
    ILUPreconditioner(
        const TMatrix<TPrecision> &arg_L,
        const TMatrix<TPrecision> &arg_U,
        const TMatrix<TPrecision> &arg_P
    ):
        m(arg_L.rows()), L(arg_L), U(arg_U), P(arg_P)
    {
        if (arg_L.rows() != arg_L.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix L");
        }
        if (arg_U.rows() != arg_U.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix U");
        }
        if (arg_P.rows() != arg_P.cols()) {
            throw std::runtime_error("ILU(L, U, P): Non square matrix P");
        }
        if (arg_L.rows() != arg_U.rows()) {
            throw std::runtime_error("ILU(L, U, P): L and U dim mismatch");
        }
        if (arg_L.rows() != arg_P.rows()) {
            throw std::runtime_error("ILU(L, U, P): L and P dim mismatch");
        }
    }

    // ILU(0) constructors
    ILUPreconditioner(const MatrixDense<TPrecision> &A):
        m(A.rows())
    {
        ilu_subrtns::ILUTriplet<MatrixDense, TPrecision> ret = (
            ilu_subrtns::construct_square_ILU_0<MatrixDense, TPrecision>(
                NoFillMatrixSparse<TPrecision>(A)
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }
    ILUPreconditioner(const NoFillMatrixSparse<TPrecision> &A):
        m(A.rows())
    {
        ilu_subrtns::ILUTriplet<NoFillMatrixSparse, TPrecision> ret = (
            ilu_subrtns::construct_square_ILU_0<NoFillMatrixSparse, TPrecision>(
                A
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    /* ILUT(tau, p), constructors tau threshold to drop and p number of entries
       to keep */
    ILUPreconditioner(
        const MatrixDense<TPrecision> &A,
        const TPrecision &tau,
        const int &p,
        const bool &to_pivot
    ):
        m(A.rows())
    {
        ilu_subrtns::ILUTriplet<MatrixDense, TPrecision> ret = (
            ilu_subrtns::construct_square_ILUTP<MatrixDense, TPrecision>(
                NoFillMatrixSparse<TPrecision>(A), tau, p, to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }
    ILUPreconditioner(
        const NoFillMatrixSparse<TPrecision> &A,
        const TPrecision &tau,
        const int &p,
        const bool &to_pivot
    ):
        m(A.rows())
    {
        ilu_subrtns::ILUTriplet<NoFillMatrixSparse, TPrecision> ret = (
            ilu_subrtns::construct_square_ILUTP<NoFillMatrixSparse, TPrecision>(
                A, tau, p, to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    TMatrix<TPrecision> get_L() const { return L; }
    TMatrix<TPrecision> get_U() const { return U; }
    TMatrix<TPrecision> get_P() const { return P; }

    Vector<TPrecision> action_inv_M(
        const Vector<TPrecision> &vec
    ) const override {
        return U.back_sub(L.frwd_sub(P*vec));
    }

    bool check_compatibility_left(const int &arg_m) const override {
        return arg_m == m;
    };
    bool check_compatibility_right(const int &arg_n) const override {
        return arg_n == m;
    };

    ILUPreconditioner<TMatrix, double> * cast_dbl_ptr() const override {
        return new ILUPreconditioner<TMatrix, double>(
            L.template cast<double>(),
            U.template cast<double>(),
            P.template cast<double>()
        );
    }

    ILUPreconditioner<TMatrix, float> * cast_sgl_ptr() const override {
        return new ILUPreconditioner<TMatrix, float>(
            L.template cast<float>(),
            U.template cast<float>(),
            P.template cast<float>()
        );
    }

    ILUPreconditioner<TMatrix, __half> * cast_hlf_ptr() const override {
        return new ILUPreconditioner<TMatrix, __half>(
            L.template cast<__half>(),
            U.template cast<__half>(),
            P.template cast<__half>()
        );
    }

};

#endif
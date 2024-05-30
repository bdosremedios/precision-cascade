#ifndef ILU_PRECONDITIONER_H
#define ILU_PRECONDITIONER_H

#include "Preconditioner.h"
#include "tools/ILU_subroutines.h"

template <template <typename> typename M, typename W>
class ILUPreconditioner: public Preconditioner<M, W>
{
private:

    int m;
    M<W> L = M<W>(cuHandleBundle());
    M<W> U = M<W>(cuHandleBundle());
    M<W> P = M<W>(cuHandleBundle());

public:

    // *** Constructors ***
    
    // ILU constructor taking premade L and U and no permutation
    ILUPreconditioner(const M<W> &arg_L, const M<W> &arg_U):
        ILUPreconditioner(
            arg_L, arg_U, M<W>::Identity(arg_L.get_cu_handles(), arg_L.rows(), arg_L.rows())
        )
    {}

    // ILU constructor taking premade L, U, and P
    ILUPreconditioner(const M<W> &arg_L, const M<W> &arg_U, const M<W> &arg_P):
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

    // ILU(0)
    ILUPreconditioner(const MatrixDense<W> &A, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<MatrixDense, W> ret = (
            ilu_subroutines::construct_square_ILU_0<MatrixDense, W>(
                NoFillMatrixSparse<W>(A), to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILUPreconditioner(const NoFillMatrixSparse<W> &A, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<NoFillMatrixSparse, W> ret = (
            ilu_subroutines::construct_square_ILU_0<NoFillMatrixSparse, W>(
                A, to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    ILUPreconditioner(const MatrixDense<W> &A, const W &tau, const int &p, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<MatrixDense, W> ret = (
            ilu_subroutines::construct_square_ILUTP<MatrixDense, W>(
                NoFillMatrixSparse<W>(A), tau, p, to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    ILUPreconditioner(const NoFillMatrixSparse<W> &A, const W &tau, const int &p, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<NoFillMatrixSparse, W> ret = (
            ilu_subroutines::construct_square_ILUTP<NoFillMatrixSparse, W>(
                A, tau, p, to_pivot
            )
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    M<W> get_L() const { return L; }
    M<W> get_U() const { return U; }
    M<W> get_P() const { return P; }

    // *** Concrete Methods ***

    Vector<W> action_inv_M(const Vector<W> &vec) const override {
        return U.back_sub(L.frwd_sub(P*vec));
    }

    bool check_compatibility_left(const int &arg_m) const override { return arg_m == m; };
    bool check_compatibility_right(const int &arg_n) const override { return arg_n == m; };

    ILUPreconditioner<M, double> * cast_dbl_ptr() const override {
        return new ILUPreconditioner<M, double>(
            L.template cast<double>(), U.template cast<double>(), P.template cast<double>()
        );
    }

    ILUPreconditioner<M, float> * cast_sgl_ptr() const override {
        return new ILUPreconditioner<M, float>(
            L.template cast<float>(), U.template cast<float>(), P.template cast<float>()
        );
    }

    ILUPreconditioner<M, __half> * cast_hlf_ptr() const override {
        return new ILUPreconditioner<M, __half>(
            L.template cast<__half>(), U.template cast<__half>(), P.template cast<__half>()
        );
    }

};

#endif
#ifndef ILU_PRECONDITIONER_H
#define ILU_PRECONDITIONER_H

#include <cmath>
#include <vector>
#include <functional>

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
    // std::function<bool (const W &curr_val, const int &row, const int &col, const W &zero_tol)> drop_rule_tau;
    // std::function<void (const int &col, const W &zero_tol, const int &m, W *U_mat, W *L_mat)> apply_drop_rule_col;

    // void construct_ILU(const M<W> &A, const W &zero_tol, const bool &pivot) {
    //     ilu::dynamic_construct_leftlook_square_ILU(
    //         zero_tol, pivot, drop_rule_tau, apply_drop_rule_col, A, U, L, P
    //     );
    // }

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
        ilu_subroutines::ILUTriplet<MatrixDense, W> ret = ilu_subroutines::construct_square_ILU_0<MatrixDense, W>(
            NoFillMatrixSparse<W>(A), to_pivot
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    // ILUT(tau, p), tau threshold to drop and p number of entries to keep
    ILUPreconditioner(const NoFillMatrixSparse<W> &A, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<NoFillMatrixSparse, W> ret = ilu_subroutines::construct_square_ILU_0<NoFillMatrixSparse, W>(
            A, to_pivot
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    ILUPreconditioner(const MatrixDense<W> &A, const W &tau, const int &p, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<MatrixDense, W> ret = ilu_subroutines::construct_square_ILU_0<MatrixDense, W>(
            NoFillMatrixSparse<W>(A), tau, p, to_pivot
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    ILUPreconditioner(const NoFillMatrixSparse<W> &A, const W &tau, const int &p, const bool &to_pivot):
        m(A.rows())
    {
        ilu_subroutines::ILUTriplet<NoFillMatrixSparse, W> ret = ilu_subroutines::construct_square_ILU_0<NoFillMatrixSparse, W>(
            A, tau, p, to_pivot
        );
        L = ret.L; U = ret.U; P = ret.P;
    }

    Vector<W> action_inv_M(const Vector<W> &vec) const override {
        return U.back_sub(L.frwd_sub(P*vec));
    }

    M<W> get_L() const { return L; }
    M<W> get_U() const { return U; }
    M<W> get_P() const { return P; }

    template <typename T>
    M<T> get_L_cast() const { return L.template cast<T>(); }

    template <typename T>
    M<T> get_U_cast() const { return U.template cast<T>(); }

    bool check_compatibility_left(const int &arg_m) const override { return arg_m == m; };
    bool check_compatibility_right(const int &arg_n) const override { return arg_n == m; };

};

#endif
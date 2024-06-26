#ifndef SOR_SOLVE_H
#define SOR_SOLVE_H

#include "../IterativeSolve.h"

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class SORSolve: public TypedIterativeSolve<TMatrix, TPrecision>
{
private:

    Scalar<TPrecision> w;
    TMatrix<TPrecision> D_wL = TMatrix<TPrecision>(cuHandleBundle());

protected:

    using TypedIterativeSolve<TMatrix, TPrecision>::typed_lin_sys_ptr;
    using TypedIterativeSolve<TMatrix, TPrecision>::typed_soln;

    void typed_iterate() override {
        typed_soln += (
            D_wL.frwd_sub(typed_lin_sys_ptr->get_b_typed() -
            typed_lin_sys_ptr->get_A_typed()*typed_soln)
        )*w;
    }
    void derived_typed_reset() override {}; // Set reset as empty function

public:

    SORSolve(
        const TypedLinearSystem_Intf<MatrixDense, TPrecision> * const arg_typed_lin_sys_ptr,
        double arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<TPrecision>(arg_w)),
        TypedIterativeSolve<MatrixDense, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr,
            arg_pkg
        )
    {

        TPrecision *h_D_wL = static_cast<TPrecision *>(
            malloc(
                typed_lin_sys_ptr->get_m() *
                typed_lin_sys_ptr->get_n() *
                sizeof(TPrecision)
            )
        );
        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            h_D_wL, typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n()
        );

        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
                if (i < j) {
                    h_D_wL[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<TPrecision>(0)
                    );
                } else if (i > j) {
                    h_D_wL[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<TPrecision>(arg_w) *
                        h_D_wL[i+j*typed_lin_sys_ptr->get_m()]
                    );
                }
            }
        }

        D_wL = MatrixDense<TPrecision>(
            typed_lin_sys_ptr->get_cu_handles(),
            h_D_wL,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n()
        );

        free(h_D_wL);

    }

    SORSolve(
        const TypedLinearSystem_Intf<NoFillMatrixSparse, TPrecision> * const arg_typed_lin_sys_ptr,
        double arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<TPrecision>(arg_w)),
        TypedIterativeSolve<NoFillMatrixSparse, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr,
            arg_pkg
        )
    {

        int *new_col_offsets = static_cast<int *>(
            malloc((typed_lin_sys_ptr->get_n()+1)*sizeof(int))
        );
        new_col_offsets[0] = 0;
        std::vector<int> new_row_indices;
        std::vector<TPrecision> new_vals;

        int *A_col_offsets = static_cast<int *>(
            malloc((typed_lin_sys_ptr->get_n()+1)*sizeof(int))
        );
        int *A_row_indices = static_cast<int *>(
            malloc(typed_lin_sys_ptr->get_nnz()*sizeof(int))
        );
        TPrecision *A_vals = static_cast<TPrecision *>(
            malloc(typed_lin_sys_ptr->get_nnz()*sizeof(TPrecision))
        );

        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            A_col_offsets, A_row_indices, A_vals,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n(),
            typed_lin_sys_ptr->get_nnz()
        );

        int D_wL_count = 0;
        for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
            int col_beg = A_col_offsets[j];
            int col_end = A_col_offsets[j+1];
            for (int offset=col_beg; offset<col_end; ++offset) {
                int row = A_row_indices[offset];
                if (row >= j) {
                    TPrecision val = A_vals[offset];
                    if (val != static_cast<TPrecision>(0.)) {
                        new_row_indices.push_back(row);
                        if (row == j) {
                            new_vals.push_back(val);
                        } else {
                            new_vals.push_back(
                                static_cast<TPrecision>(arg_w)*val
                            );
                        }
                        ++D_wL_count;
                    }
                }
            }
            new_col_offsets[j+1] = D_wL_count;
        }

        D_wL = NoFillMatrixSparse<TPrecision>(
            typed_lin_sys_ptr->get_cu_handles(),
            new_col_offsets, &new_row_indices[0], &new_vals[0],
            typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n(), D_wL_count
        );

        free(new_col_offsets);

        free(A_col_offsets);
        free(A_row_indices);
        free(A_vals);

    }

    // Forbid rvalue instantiation
    SORSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const,
        double,
        const SolveArgPkg &&
    ) = delete;

};

}

#endif
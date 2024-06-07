#ifndef JACOBI_SOLVE_H
#define JACOBI_SOLVE_H

#include "../IterativeSolve.h"

template <template <typename> typename TMatrix, typename TPrecision>
class JacobiSolve: public TypedIterativeSolve<TMatrix, TPrecision>
{
private:
    
    TMatrix<TPrecision> D_inv = TMatrix<TPrecision>(cuHandleBundle());

protected:

    using TypedIterativeSolve<TMatrix, TPrecision>::typed_lin_sys_ptr;
    using TypedIterativeSolve<TMatrix, TPrecision>::typed_soln;

    void typed_iterate() override {
        typed_soln += D_inv*(
            typed_lin_sys_ptr->get_b_typed() -
            typed_lin_sys_ptr->get_A_typed()*typed_soln
        );
    }
    void derived_typed_reset() override {}; // Set reset as empty function

public:

    JacobiSolve(
        const TypedLinearSystem_Intf<MatrixDense, TPrecision> * const arg_typed_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ):
        TypedIterativeSolve<MatrixDense, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr,
            arg_pkg
        )
    {

        TPrecision *h_mat = static_cast<TPrecision *>(
            malloc(
                typed_lin_sys_ptr->get_m() *
                typed_lin_sys_ptr->get_n() *
                sizeof(TPrecision)
            )
        );
        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            h_mat, typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n()
        );

        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
                if (i == j) {
                    h_mat[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<TPrecision>(1) /
                        h_mat[i+j*typed_lin_sys_ptr->get_m()]
                    );
                } else {
                    h_mat[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<TPrecision>(0)
                    );
                }
            }
        }

        D_inv = MatrixDense<TPrecision>(
            typed_lin_sys_ptr->get_cu_handles(),
            h_mat,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n()
        );

        free(h_mat);

    }

    JacobiSolve(
        const TypedLinearSystem_Intf<NoFillMatrixSparse, TPrecision> * const arg_typed_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ):
        TypedIterativeSolve<NoFillMatrixSparse, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr,
            arg_pkg
        )
    {

        int *new_col_offsets = static_cast<int *>(
            malloc((typed_lin_sys_ptr->get_m()+1)*sizeof(int))
        );
        int *new_row_indices = static_cast<int *>(
            malloc(typed_lin_sys_ptr->get_m()*sizeof(int))
        );
        TPrecision *new_vals = static_cast<TPrecision *>(
            malloc(typed_lin_sys_ptr->get_m()*sizeof(TPrecision))
        );

        for (int i=0; i<typed_lin_sys_ptr->get_m()+1; ++i) {
            new_col_offsets[i] = i;
        }
        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            new_row_indices[i] = i;
        }
        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            new_vals[i] = (
                1./typed_lin_sys_ptr->get_A().get_elem(i, i).get_scalar()
            );
        }

        D_inv = NoFillMatrixSparse<TPrecision>(
            typed_lin_sys_ptr->get_cu_handles(),
            new_col_offsets, new_row_indices, new_vals,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n(),
            typed_lin_sys_ptr->get_m()
        );

        free(new_col_offsets);
        free(new_row_indices);
        free(new_vals);

    }

    // Forbid rvalue instantiation
    JacobiSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const,
        const SolveArgPkg &&
    ) = delete;

};

#endif
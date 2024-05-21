#ifndef JACOBI_SOLVE_H
#define JACOBI_SOLVE_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class JacobiSolve: public TypedIterativeSolve<M, T>
{
private:
    
    M<T> D_inv = M<T>(cuHandleBundle());

protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys_ptr;
    using TypedIterativeSolve<M, T>::typed_soln;

    // *** Helper Methods ***
    void typed_iterate() override {
        typed_soln += D_inv*(typed_lin_sys_ptr->get_b_typed()-typed_lin_sys_ptr->get_A_typed()*typed_soln);
    }
    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** Constructors ***
    JacobiSolve(
        TypedLinearSystem<MatrixDense, T> * const arg_typed_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ):
        TypedIterativeSolve<MatrixDense, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, arg_pkg)
    {

        T *h_mat = static_cast<T *>(malloc(typed_lin_sys_ptr->get_m()*typed_lin_sys_ptr->get_n()*sizeof(T)));
        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            h_mat, typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n()
        );

        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
                if (i == j) {
                    h_mat[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<T>(1)/h_mat[i+j*typed_lin_sys_ptr->get_m()]
                    );
                } else {
                    h_mat[i+j*typed_lin_sys_ptr->get_m()] = static_cast<T>(0);
                }
            }
        }

        D_inv = MatrixDense<T>(
            typed_lin_sys_ptr->get_cu_handles(),
            h_mat,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n()
        );

        free(h_mat);

    }

    JacobiSolve(
        TypedLinearSystem<NoFillMatrixSparse, T> * const arg_typed_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ):
        TypedIterativeSolve<NoFillMatrixSparse, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, arg_pkg)
    {

        int *new_col_offsets = static_cast<int *>(malloc((typed_lin_sys_ptr->get_m()+1)*sizeof(int)));
        int *new_row_indices = static_cast<int *>(malloc(typed_lin_sys_ptr->get_m()*sizeof(int)));
        T *new_vals = static_cast<T *>(malloc(typed_lin_sys_ptr->get_m()*sizeof(T)));

        for (int i=0; i<typed_lin_sys_ptr->get_m()+1; ++i) { new_col_offsets[i] = i; }
        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) { new_row_indices[i] = i; }
        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            new_vals[i] = 1./typed_lin_sys_ptr->get_A().get_elem(i, i).get_scalar();
        }

        D_inv = NoFillMatrixSparse<T>(
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
    JacobiSolve(TypedLinearSystem<M, T> * const, const SolveArgPkg &&) = delete;

};

#endif
#ifndef SOR_SOLVE_H
#define SOR_SOLVE_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class SORSolve: public TypedIterativeSolve<M, T>
{
private:

    Scalar<T> w;
    M<T> D_wL = M<T>(cuHandleBundle());

protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys_ptr;
    using TypedIterativeSolve<M, T>::typed_soln;

    // *** Helper Methods ***
    void typed_iterate() override {
        typed_soln += (
            (D_wL.frwd_sub(typed_lin_sys_ptr->get_b_typed()-typed_lin_sys_ptr->get_A_typed()*typed_soln))*w
        );
    }
    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** Constructors ***
    SORSolve(
        TypedLinearSystem<MatrixDense, T> * const arg_typed_lin_sys_ptr,
        double arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<MatrixDense, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, arg_pkg)
    {

        T *h_D_wL = static_cast<T *>(
            malloc(typed_lin_sys_ptr->get_m()*typed_lin_sys_ptr->get_n()*sizeof(T))
        );
        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            h_D_wL, typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n()
        );

        for (int i=0; i<typed_lin_sys_ptr->get_m(); ++i) {
            for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
                if (i < j) {
                    h_D_wL[i+j*typed_lin_sys_ptr->get_m()] = static_cast<T>(0);
                } else if (i > j) {
                    h_D_wL[i+j*typed_lin_sys_ptr->get_m()] = (
                        static_cast<T>(arg_w)*h_D_wL[i+j*typed_lin_sys_ptr->get_m()]
                    );
                }
            }
        }

        D_wL = MatrixDense<T>(
            typed_lin_sys_ptr->get_cu_handles(),
            h_D_wL,
            typed_lin_sys_ptr->get_m(),
            typed_lin_sys_ptr->get_n()
        );

        free(h_D_wL);

    }

    SORSolve(
        TypedLinearSystem<NoFillMatrixSparse, T> * const arg_typed_lin_sys_ptr,
        double arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<NoFillMatrixSparse, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, arg_pkg)
    {

        int *new_col_offsets = static_cast<int *>(malloc((typed_lin_sys_ptr->get_n()+1)*sizeof(int)));
        new_col_offsets[0] = 0;
        std::vector<int> new_row_indices;
        std::vector<T> new_vals;

        int *A_col_offsets = static_cast<int *>(malloc((typed_lin_sys_ptr->get_n()+1)*sizeof(int)));
        int *A_row_indices = static_cast<int *>(malloc(typed_lin_sys_ptr->get_nnz()*sizeof(int)));
        T *A_vals = static_cast<T *>(malloc(typed_lin_sys_ptr->get_nnz()*sizeof(T)));

        typed_lin_sys_ptr->get_A_typed().copy_data_to_ptr(
            A_col_offsets, A_row_indices, A_vals,
            typed_lin_sys_ptr->get_m(), typed_lin_sys_ptr->get_n(), typed_lin_sys_ptr->get_nnz()
        );

        int D_wL_count = 0;
        for (int j=0; j<typed_lin_sys_ptr->get_n(); ++j) {
            int col_beg = A_col_offsets[j];
            int col_end = A_col_offsets[j+1];
            for (int offset=col_beg; offset<col_end; ++offset) {
                int row = A_row_indices[offset];
                if (row >= j) {
                    T val = A_vals[offset];
                    if (val != static_cast<T>(0.)) {
                        new_row_indices.push_back(row);
                        if (row == j) {
                            new_vals.push_back(val);
                        } else {
                            new_vals.push_back(static_cast<T>(arg_w)*val);
                        }
                        ++D_wL_count;
                    }
                }
            }
            new_col_offsets[j+1] = D_wL_count;
        }

        D_wL = NoFillMatrixSparse<T>(
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
    SORSolve(TypedLinearSystem<M, T> * const, double, const SolveArgPkg &&) = delete;

};

#endif
#ifndef JACOBI_PRECONDITIONER_H
#define JACOBI_PRECONDITIONER_H

#include "MatrixInversePreconditioner.h"

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class JacobiPreconditioner:
    public MatrixInversePreconditioner<TMatrix, TPrecision>
{
private:

    MatrixDense<TPrecision> construct_dense_jacobi(
        MatrixDense<TPrecision> const &arg_A
    ) {
        
        TPrecision * h_mat = static_cast<TPrecision *>(
            malloc(arg_A.rows()*arg_A.cols()*sizeof(TPrecision))
        );
        arg_A.copy_data_to_ptr(h_mat, arg_A.rows(), arg_A.cols());
        
        for (int j=0; j<arg_A.cols(); ++j) {
            for (int i=0; i<arg_A.rows(); ++i) {
                if (i == j) {
                    if (h_mat[i+arg_A.rows()*j] != static_cast<TPrecision>(0.)) {
                        h_mat[i+arg_A.rows()*j] = (
                            static_cast<TPrecision>(1.)/h_mat[i+arg_A.rows()*j]
                        );
                    } else {
                        free(h_mat);
                        throw std::runtime_error(
                            "JacobiPreconditioner: zero in "
                            "construct_dense_jacobi encountered at "
                            "(" + std::to_string(i) + ", " +
                            std::to_string(j) + ")"
                        );
                    }
                } else {
                    h_mat[i+arg_A.rows()*j] = static_cast<TPrecision>(0.);
                }
            }
        }

        MatrixDense<TPrecision> ret_mat(
            arg_A.get_cu_handles(),
            h_mat,
            arg_A.rows(),
            arg_A.cols()
        );

        free(h_mat);

        return ret_mat;

    }

    NoFillMatrixSparse<TPrecision> construct_sparse_jacobi(
        NoFillMatrixSparse<TPrecision> const &arg_A
    ) {

        int * h_col_offsets = static_cast<int *>(
            malloc((arg_A.cols()+1)*sizeof(int))
        );
        int * h_row_indices = static_cast<int *>(
            malloc(arg_A.non_zeros()*sizeof(int))
        );
        TPrecision * h_vals = static_cast<TPrecision *>(
            malloc(arg_A.non_zeros()*sizeof(TPrecision))
        );
        arg_A.copy_data_to_ptr(
            h_col_offsets, h_row_indices, h_vals,
            arg_A.rows(), arg_A.cols(), arg_A.non_zeros()
        );

        int smaller_dim = std::min(arg_A.rows(), arg_A.cols());

        int * new_h_col_offsets = static_cast<int *>(
            malloc((arg_A.cols()+1)*sizeof(int))
        );
        int * new_h_row_indices = static_cast<int *>(
            malloc(smaller_dim*sizeof(int))
        );
        TPrecision * new_h_vals = static_cast<TPrecision *>(
            malloc(smaller_dim*sizeof(TPrecision))
        );

        for (int j=0; j<smaller_dim; ++j) {

            new_h_col_offsets[j] = j;
            new_h_row_indices[j] = j;

            TPrecision diag_val = static_cast<TPrecision>(0.);
            for (
                int offset=h_col_offsets[j];
                offset<h_col_offsets[j+1];
                ++offset
            ) {
                if (h_row_indices[offset] == j) {
                    diag_val = h_vals[offset];
                    break;
                }
            }

            if (diag_val != static_cast<TPrecision>(0.)) {
                new_h_vals[j] = static_cast<TPrecision>(1.)/diag_val;
            } else {
                free(h_col_offsets);
                free(h_row_indices);
                free(h_vals);
                throw std::runtime_error(
                    "JacobiPreconditioner: zero in "
                    "construct_sparse_jacobi encountered at "
                    "(" + std::to_string(j) + ", " +
                    std::to_string(j) + ")"
                );
            }

        }
        for (int j=smaller_dim; j<arg_A.cols()+1; ++j) {
            new_h_col_offsets[j] = smaller_dim;
        }

        NoFillMatrixSparse<TPrecision> ret_mat(
            arg_A.get_cu_handles(),
            new_h_col_offsets, new_h_row_indices, new_h_vals,
            arg_A.rows(), arg_A.cols(), smaller_dim
        );

        free(h_col_offsets);
        free(h_row_indices);
        free(h_vals);

        free(new_h_col_offsets);
        free(new_h_row_indices);
        free(new_h_vals);

        return ret_mat;

    }

public:

    JacobiPreconditioner(MatrixDense<TPrecision> const &arg_A):
        MatrixInversePreconditioner<MatrixDense, TPrecision>(
            construct_dense_jacobi(arg_A)
        )
    {}

    JacobiPreconditioner(NoFillMatrixSparse<TPrecision> const &arg_A):
        MatrixInversePreconditioner<NoFillMatrixSparse, TPrecision>(
            construct_sparse_jacobi(arg_A)
        )
    {}

};

}

#endif
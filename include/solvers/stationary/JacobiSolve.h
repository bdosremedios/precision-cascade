#ifndef JACOBI_SOLVE_H
#define JACOBI_SOLVE_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class JacobiSolve: public TypedIterativeSolve<M, T>
{
private:
    
    M<T> D_inv = M<T>(NULL);

protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::typed_soln;

    void typed_iterate() override {
        typed_soln += D_inv*(typed_lin_sys.get_b_typed()-typed_lin_sys.get_A_typed()*typed_soln);
    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** Constructors ***
    JacobiSolve(const TypedLinearSystem<M, T> &arg_typed_lin_sys, const SolveArgPkg &arg_pkg):
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys, arg_pkg)
    {

        T *h_mat = static_cast<T *>(malloc(typed_lin_sys.get_m()*typed_lin_sys.get_n()*sizeof(T)));
        typed_lin_sys.get_A_typed().copy_data_to_ptr(h_mat, typed_lin_sys.get_m(), typed_lin_sys.get_n());
        for (int i=0; i<typed_lin_sys.get_m(); ++i) {
            for (int j=0; j<typed_lin_sys.get_n(); ++j) {
                if (i == j) {
                    h_mat[i+j*typed_lin_sys.get_m()] = static_cast<T>(1)/h_mat[i+j*typed_lin_sys.get_m()];
                } else {
                    h_mat[i+j*typed_lin_sys.get_m()] = static_cast<T>(0);
                }
            }
        }

        D_inv = M<T>(
            arg_typed_lin_sys.get_A_typed().get_handle(),
            h_mat,
            typed_lin_sys.get_m(),
            typed_lin_sys.get_n()
        );

        free(h_mat);

    }

};

#endif
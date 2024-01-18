#ifndef SOR_SOLVE_H
#define SOR_SOLVE_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T>
class SORSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::typed_soln;

    T w;
    M<T> D_wL = M<T>(NULL);
    M<T> L_wL_U = M<T>(NULL);

    void typed_iterate() override {
        typed_soln = D_wL.frwd_sub(typed_lin_sys.get_b_typed()-L_wL_U*typed_soln);
    }

    void derived_typed_reset() override {}; // Set reset as empty function

public:

    // *** Constructors ***
    SORSolve(
        const TypedLinearSystem<M, T> &arg_typed_lin_sys,
        double arg_w,
        const SolveArgPkg &arg_pkg
    ):
        w(static_cast<T>(arg_w)),
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys, arg_pkg)
    {

        T *h_D_wL = static_cast<T *>(malloc(typed_lin_sys.get_m()*typed_lin_sys.get_n()*sizeof(T)));
        typed_lin_sys.get_A_typed().copy_data_to_ptr(h_D_wL, typed_lin_sys.get_m(), typed_lin_sys.get_n());
        for (int i=0; i<typed_lin_sys.get_m(); ++i) {
            for (int j=0; j<typed_lin_sys.get_n(); ++j) {
                if (i < j) {
                    h_D_wL[i+j*typed_lin_sys.get_m()] = static_cast<T>(0);
                } else if (i > j) {
                    h_D_wL[i+j*typed_lin_sys.get_m()] = w*h_D_wL[i+j*typed_lin_sys.get_m()];
                }
            }
        }

        D_wL = M<T>(
            arg_typed_lin_sys.get_A_typed().get_handle(),
            h_D_wL,
            typed_lin_sys.get_m(),
            typed_lin_sys.get_n()
        );

        free(h_D_wL);

        T *h_L_wL_U = static_cast<T *>(malloc(typed_lin_sys.get_m()*typed_lin_sys.get_n()*sizeof(T)));
        typed_lin_sys.get_A_typed().copy_data_to_ptr(h_L_wL_U, typed_lin_sys.get_m(), typed_lin_sys.get_n());
        for (int i=0; i<typed_lin_sys.get_m(); ++i) {
            for (int j=0; j<typed_lin_sys.get_n(); ++j) {
                if (i == j) {
                    h_L_wL_U[i+j*typed_lin_sys.get_m()] = static_cast<T>(0);
                } else if (i > j) {
                    h_L_wL_U[i+j*typed_lin_sys.get_m()] = (
                        (static_cast<T>(1)-w)*h_L_wL_U[i+j*typed_lin_sys.get_m()]
                    );
                }
            }
        }

        L_wL_U = M<T>(
            arg_typed_lin_sys.get_A_typed().get_handle(),
            h_L_wL_U,
            typed_lin_sys.get_m(),
            typed_lin_sys.get_n()
        );

        free(h_L_wL_U);

    }

    // Forbid rvalue instantiation
    SORSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &) = delete;
    SORSolve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&) = delete;
    SORSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&) = delete;

};

#endif
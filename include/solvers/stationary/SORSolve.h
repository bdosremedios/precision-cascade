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

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::typed_soln;

    // *** Helper Methods ***
    void typed_iterate() override {
        typed_soln += (D_wL.frwd_sub(typed_lin_sys.get_b_typed()-typed_lin_sys.get_A_typed()*typed_soln))*w;
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
                    h_D_wL[i+j*typed_lin_sys.get_m()] = static_cast<T>(arg_w)*h_D_wL[i+j*typed_lin_sys.get_m()];
                }
            }
        }

        D_wL = M<T>(
            arg_typed_lin_sys.get_cu_handles(),
            h_D_wL,
            typed_lin_sys.get_m(),
            typed_lin_sys.get_n()
        );

        free(h_D_wL);

    }

    // Forbid rvalue instantiation
    SORSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &) = delete;
    SORSolve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&) = delete;
    SORSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&) = delete;

};

#endif
#ifndef FP_GMRES_IR_SOLVE_H
#define FP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

template <template <typename> typename M, typename T, typename W=T>
class FP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    void set_inner_solve() {

        IR_inner_typed_lin_sys.set_b(this->curr_res);

        this->inner_solver = std::make_shared<GMRESSolve<M, T>>(
            IR_inner_typed_lin_sys,
            basis_zero_tol,
            this->inner_solve_arg_pkg,
            inner_precond_arg_pkg
        );

    }

protected:

    // *** Const Attributes ***
    const double basis_zero_tol;
    const PrecondArgPkg<M, W> inner_precond_arg_pkg;

    // *** Mutable Attributes ***
    Mutb_TypedLinearSystem<M, T> IR_inner_typed_lin_sys;

    // *** Virtual Abstract Methods ***
    void initialize_inner_outer_solver() override { set_inner_solve(); }
    void outer_iterate_setup() override { set_inner_solve(); }
    void derived_generic_reset() override { set_inner_solve(); }
    
public:

    using IterativeRefinement<M>::lin_sys;

    // *** Constructors ***
    FP_GMRES_IR_Solve(
        const TypedLinearSystem<M, T> &arg_outer_typed_lin_sys,
        double arg_basis_zero_tol,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<M, W> &arg_inner_precond_arg_pkg = PrecondArgPkg<M, W>()
    ):
        basis_zero_tol(arg_basis_zero_tol),
        inner_precond_arg_pkg(arg_inner_precond_arg_pkg),
        IterativeRefinement<M>(arg_outer_typed_lin_sys, arg_solve_arg_pkg),
        IR_inner_typed_lin_sys(arg_outer_typed_lin_sys.get_A(), this->curr_res)

    {
        initialize_inner_outer_solver();
    }

    // Forbid rvalue instantiation
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;

};

#endif
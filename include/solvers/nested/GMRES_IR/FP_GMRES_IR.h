#ifndef FP_GMRES_IR_SOLVE_H
#define FP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

template <template <typename> typename M, typename T, typename W=T>
class FP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    void set_inner_solve() {

        mutrhs_innerlinsys_ptr.set_rhs(this->curr_res);

        this->inner_solver = std::make_shared<GMRESSolve<M, T>>(
            &mutrhs_innerlinsys_ptr,
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
    TypedLinearSystem_MutAddlRHS<M, T> mutrhs_innerlinsys_ptr;

    // *** Virtual Abstract Methods ***
    void initialize_inner_outer_solver() override { set_inner_solve(); }
    void outer_iterate_setup() override { set_inner_solve(); }
    void derived_generic_reset() override { set_inner_solve(); }
    
public:

    using IterativeRefinement<M>::gen_lin_sys_ptr;

    // *** Constructors ***
    FP_GMRES_IR_Solve(
        const TypedLinearSystem<M, T> * const arg_outer_typed_lin_sys_ptr,
        double arg_basis_zero_tol,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<M, W> &arg_inner_precond_arg_pkg = PrecondArgPkg<M, W>()
    ):
        basis_zero_tol(arg_basis_zero_tol),
        inner_precond_arg_pkg(arg_inner_precond_arg_pkg),
        IterativeRefinement<M>(arg_outer_typed_lin_sys_ptr->get_gen_lin_sys_ptr(), arg_solve_arg_pkg),
        mutrhs_innerlinsys_ptr(arg_outer_typed_lin_sys_ptr, this->curr_res)

    {
        initialize_inner_outer_solver();
    }

    // Forbid rvalue instantiation
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> * const, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> * const, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    FP_GMRES_IR_Solve(const TypedLinearSystem<M, T> * const, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;

};

#endif
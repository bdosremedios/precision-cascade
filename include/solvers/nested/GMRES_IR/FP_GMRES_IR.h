#ifndef FP_GMRES_IR_SOLVE_H
#define FP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class FP_GMRES_IR_Solve: public IterativeRefinement<TMatrix>
{
private:

    void set_inner_solve() {

        mutrhs_innerlinsys_ptr.set_rhs(this->curr_res);

        this->inner_solver = std::make_shared<GMRESSolve<TMatrix, TPrecision>>(
            &mutrhs_innerlinsys_ptr,
            basis_zero_tol,
            this->inner_solve_arg_pkg,
            inner_precond_arg_pkg
        );

    }

protected:

    const double basis_zero_tol;
    const PrecondArgPkg<TMatrix, TPrecision> inner_precond_arg_pkg;

    TypedLinearSystem_MutAddlRHS<TMatrix, TPrecision> mutrhs_innerlinsys_ptr;

    void initialize_inner_outer_solver() override { set_inner_solve(); }
    void outer_iterate_setup() override { set_inner_solve(); }
    void derived_generic_reset() override { set_inner_solve(); }
    
public:

    using IterativeRefinement<TMatrix>::gen_lin_sys_ptr;

    FP_GMRES_IR_Solve(
        const TypedLinearSystem<TMatrix, TPrecision> * const arg_outer_typed_lin_sys_ptr,
        double arg_basis_zero_tol,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, TPrecision> arg_inner_precond_arg_pkg = (
            PrecondArgPkg<TMatrix, TPrecision>()
        )
    ):
        basis_zero_tol(arg_basis_zero_tol),
        inner_precond_arg_pkg(arg_inner_precond_arg_pkg),
        IterativeRefinement<TMatrix>(
            arg_outer_typed_lin_sys_ptr->get_gen_lin_sys_ptr(),
            arg_solve_arg_pkg
        ),
        mutrhs_innerlinsys_ptr(arg_outer_typed_lin_sys_ptr, this->curr_res)

    {
        initialize_inner_outer_solver();
    }

    // Forbid rvalue instantiation
    FP_GMRES_IR_Solve(
        const TypedLinearSystem<TMatrix, TPrecision> * const,
        double,
        const SolveArgPkg &&,
        const PrecondArgPkg<TMatrix, TPrecision>
    ) = delete;

};

}

#endif
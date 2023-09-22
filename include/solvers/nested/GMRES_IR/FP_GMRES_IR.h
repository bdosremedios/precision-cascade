#ifndef FP_GMRES_IR_SOLVE_H
#define FP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../krylov/GMRES.h"

template <template <typename> typename M, typename T, typename W=T>
class FP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    // *** PRIVATE HELPER METHODS ***

    void set_inner_solve() {
        this->inner_solver = make_shared<GMRESSolve<M, T>>(
            this->A,
            this->curr_res,
            basis_zero_tol,
            this->inner_solve_arg_pkg,
            inner_precond_arg_pkg
        );
    }

protected:

    // *** PROTECTED ATTRIBUTES ***
    double basis_zero_tol;
    PrecondArgPkg<M, W> inner_precond_arg_pkg;

    // *** PROTECTED OVERRIDE METHODS ***

    // Initialize inner outer solver;
    void initialize_inner_outer_solver() override { set_inner_solve(); }

    // Specify inner_solver for outer_iterate_calc and setup
    void outer_iterate_setup() override { set_inner_solve(); }

    void derived_generic_reset() override {} // Explicitly set as empty function
    
    public:

    // *** CONSTRUCTORS ***

    FP_GMRES_IR_Solve(
        M<double> const &arg_A,
        MatrixVector<double> const &arg_b,
        double const &arg_basis_zero_tol,
        SolveArgPkg const &arg_solve_arg_pkg,
        PrecondArgPkg<M, W> const &arg_inner_precond_arg_pkg = PrecondArgPkg<M, W>()
    ):
        basis_zero_tol(arg_basis_zero_tol),
        inner_precond_arg_pkg(arg_inner_precond_arg_pkg),
        IterativeRefinement<M>(arg_A, arg_b, arg_solve_arg_pkg)
    {
        initialize_inner_outer_solver();
    }

};

#endif
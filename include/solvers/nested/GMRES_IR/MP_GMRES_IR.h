#ifndef MP_GMRES_IR_SOLVE_H
#define MP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../krylov/GMRES.h"

class MP_GMRES_IR_Solve: public IterativeRefinement {

    private:

        // *** PRIVATE HELPER METHODS ***

        template <typename T>
        void set_inner_solve() {
            inner_solver = make_shared<GMRESSolve<T>>(
                A, curr_res,
                basis_zero_tol,
                inner_solve_arg_pkg
            );
        }

    protected:

        // *** PROTECTED CONSTANTS ***

        const int HLF_PHASE = 0;
        const int SGL_PHASE = 1;
        const int DBL_PHASE = 2;
        const int INIT_PHASE = HLF_PHASE;

        // *** PROTECTED ATTRIBUTES ***

        double basis_zero_tol;
        int cascade_phase;

        // *** PROTECTED ABSTRACT METHODS ***

        // Determine which phase should be used based on current phase and
        // current convergence progress
        virtual void determine_phase() = 0;

        // *** PROTECTED OVERRIDE METHODS ***

        // Initialize inner outer solver;
        void initialize_inner_outer_solver() override { set_inner_solve<half>(); }

        // Specify inner_solver for outer_iterate_calc and setup
        void outer_iterate_setup() override {
            determine_phase();
            if (cascade_phase == HLF_PHASE) { set_inner_solve<half>(); }
            else if (cascade_phase == SGL_PHASE) { set_inner_solve<float>(); }
            else { set_inner_solve<double>(); }
        }

        void derived_generic_reset() override {
            cascade_phase = INIT_PHASE;
            initialize_inner_outer_solver();
        }
    
    public:

        // *** CONSTRUCTORS ***

        MP_GMRES_IR_Solve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            double const &arg_basis_zero_tol,
            SolveArgPkg const &arg_solve_arg_pkg
        ):
            basis_zero_tol(arg_basis_zero_tol),
            IterativeRefinement(arg_A, arg_b, arg_solve_arg_pkg)
        {
            cascade_phase = INIT_PHASE;
            initialize_inner_outer_solver();
        }

};

class SimpleConstantThreshold : public MP_GMRES_IR_Solve {

    protected:

        // *** PROTECTED CONSTANTS ***
    
        const double tol_hlf = pow(10, -02);
        const double tol_sgl = pow(10, -05);
        const double tol_dbl = pow(10, -10);

        // *** PROTECTED OVERRIDE METHODS ***

        void determine_phase() override {
            if (cascade_phase == HLF_PHASE) {
                if ((this->get_relres() <= tol_hlf)) { cascade_phase = SGL_PHASE; }
            } else if (cascade_phase == SGL_PHASE) {
                if ((this->get_relres() <= tol_sgl)) { cascade_phase = DBL_PHASE; }
            } else {
                ;
            }
        }

        using MP_GMRES_IR_Solve::MP_GMRES_IR_Solve;

};

#endif
#ifndef MP_GMRES_IR_SOLVE_H
#define MP_GMRES_IR_SOLVE_H

#include <cuda_fp16.h>

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

template <template <typename> typename M>
class MP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    Mutb_TypedLinearSystem<M, __half> IR_inner_lin_sys_hlf;
    Mutb_TypedLinearSystem<M, float> IR_inner_lin_sys_sgl;
    Mutb_TypedLinearSystem<M, double> IR_inner_lin_sys_dbl;

    // *** Helper Methods ***
    template <typename T>
    void setup_inner_solve() {

        if (std::is_same<T, __half>::value) {

            IR_inner_lin_sys_hlf.set_b(this->curr_res);
            this->inner_solver = std::make_shared<GMRESSolve<M, __half>>(
                IR_inner_lin_sys_hlf,
                u_hlf,
                this->inner_solve_arg_pkg
            );

        } else if (std::is_same<T, float>::value) {

            IR_inner_lin_sys_sgl.set_b(this->curr_res);
            this->inner_solver = std::make_shared<GMRESSolve<M, float>>(
                IR_inner_lin_sys_sgl,
                u_sgl,
                this->inner_solve_arg_pkg
            );

        } else if (std::is_same<T, double>::value) {

            IR_inner_lin_sys_dbl.set_b(this->curr_res);
            this->inner_solver = std::make_shared<GMRESSolve<M, double>>(
                IR_inner_lin_sys_dbl,
                u_dbl,
                this->inner_solve_arg_pkg
            );

        } else {

            throw std::runtime_error(
                "MP_GMRES_IR_Solve: Invalid type T used in call to setup_inner_solve<T>"
            );

        }

    }

    void choose_phase_solver() {

        if (cascade_phase == HLF_PHASE) {

            setup_inner_solve<__half>();
    
        } else if (cascade_phase == SGL_PHASE) {

            setup_inner_solve<float>();

        } else if (cascade_phase == DBL_PHASE) {
            
            setup_inner_solve<double>();
            
        } else {

            throw std::runtime_error(
                "MP_GMRES_IR_Solve: Invalid cascade_phase in MP_GMRES_IR_Solver"
            );

        }

    }

protected:

    // *** Constants ***
    const double u_hlf = pow(2, -10);
    const double u_sgl = pow(2, -23);
    const double u_dbl = pow(2, -52);

    const static int HLF_PHASE = 0;
    const static int SGL_PHASE = 1;
    const static int DBL_PHASE = 2;
    inline const static int INIT_PHASE = HLF_PHASE;

    // *** Attributes ***
    int cascade_phase;

    // *** Virtual Abstract Methods ***

    // Determine which phase should be used based on current phase and
    // current convergence progress
    virtual void determine_phase() = 0;

    // *** Concrete Override Methods ***

    // Initialize inner outer solver in __half phase
    void initialize_inner_outer_solver() override {
        setup_inner_solve<__half>();
    }

    // Specify inner_solver for outer_iterate_calc and setup
    void outer_iterate_setup() override {
        determine_phase();
        choose_phase_solver();
    }

    void derived_generic_reset() override {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }
    
public:

    // *** Constructors ***
    MP_GMRES_IR_Solve(
        const GenericLinearSystem<M> &arg_lin_sys,
        const SolveArgPkg &arg_solve_arg_pkg
    ):
        IterativeRefinement<M>(arg_lin_sys, arg_solve_arg_pkg),
        IR_inner_lin_sys_hlf(arg_lin_sys.get_A(), this->curr_res),
        IR_inner_lin_sys_sgl(arg_lin_sys.get_A(), this->curr_res),
        IR_inner_lin_sys_dbl(arg_lin_sys.get_A(), this->curr_res)
    {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }

    // Forbid rvalue instantiation
    MP_GMRES_IR_Solve(const GenericLinearSystem<M> &, const SolveArgPkg &&) = delete;
    MP_GMRES_IR_Solve(const GenericLinearSystem<M> &&, const SolveArgPkg &) = delete;
    MP_GMRES_IR_Solve(const GenericLinearSystem<M> &&, const SolveArgPkg &&) = delete;

};

template <template <typename> typename M>
class SimpleConstantThreshold : public MP_GMRES_IR_Solve<M>
{
protected:

    // *** Constants ***
    const double tol_hlf = pow(10, -02);
    const double tol_sgl = pow(10, -05);
    const double tol_dbl = pow(10, -10);

    // *** Concrete Override Methods ***
    void determine_phase() override {
        if (this->cascade_phase == this->HLF_PHASE) {
            if ((this->get_relres() <= tol_hlf)) { this->cascade_phase = this->SGL_PHASE; }
        } else if (this->cascade_phase == this->SGL_PHASE) {
            if ((this->get_relres() <= tol_sgl)) { this->cascade_phase = this->DBL_PHASE; }
        } else {
            ;
        }
    }

    using MP_GMRES_IR_Solve<M>::MP_GMRES_IR_Solve;

};

#endif
#ifndef MP_GMRES_IR_SOLVE_H
#define MP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../krylov/GMRES.h"

template <template <typename> typename M>
class MP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    // *** PRIVATE HELPER METHODS ***

    template <typename T>
    void set_GMRES_inner_solve() {
        if (std::is_same<T, half>::value) {
            this->inner_solver = make_shared<GMRESSolve<M, T>>(
                this->A, this->curr_res, u_hlf, this->inner_solve_arg_pkg
            );
        } else if (std::is_same<T, float>::value) {
            this->inner_solver = make_shared<GMRESSolve<M, T>>(
                this->A, this->curr_res, u_sgl, this->inner_solve_arg_pkg
            );
        } else if (std::is_same<T, double>::value) {
            this->inner_solver = make_shared<GMRESSolve<M, T>>(
                this->A, this->curr_res, u_dbl, this->inner_solve_arg_pkg
            );
        } else {
            throw runtime_error("Invalid type T used in call to set_GMRES_inner_solve<T>");
        }
    }

    void choose_phase_solver() {
        if (cascade_phase == HLF_PHASE) { set_GMRES_inner_solve<half>(); }
        else if (cascade_phase == SGL_PHASE) { set_GMRES_inner_solve<float>(); }
        else if (cascade_phase == DBL_PHASE) { set_GMRES_inner_solve<double>(); }
        else { throw runtime_error("Invalid cascade_phase in MP_GMRES_IR_Solver"); }
    }

protected:

    // *** PROTECTED CONSTANTS ***

    const double u_hlf = pow(2, -10);
    const double u_sgl = pow(2, -23);
    const double u_dbl = pow(2, -52);

    const static int HLF_PHASE;
    const static int SGL_PHASE;
    const static int DBL_PHASE;
    const static int INIT_PHASE;

    // *** PROTECTED ATTRIBUTES ***
    int cascade_phase;

    // *** PROTECTED ABSTRACT METHODS ***

    // Determine which phase should be used based on current phase and
    // current convergence progress
    virtual void determine_phase() = 0;

    // *** PROTECTED OVERRIDE METHODS ***

    // Initialize inner outer solver;
    void initialize_inner_outer_solver() override { set_GMRES_inner_solve<half>(); }

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

    // *** CONSTRUCTORS ***

    MP_GMRES_IR_Solve(
        M<double> const &arg_A,
        MatrixVector<double> const &arg_b,
        SolveArgPkg const &arg_solve_arg_pkg
    ):
        IterativeRefinement<M>(arg_A, arg_b, arg_solve_arg_pkg)
    {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }

};

template <template <typename> typename M>
const int MP_GMRES_IR_Solve<M>::HLF_PHASE = 0;

template <template <typename> typename M>
const int MP_GMRES_IR_Solve<M>::SGL_PHASE = 1;

template <template <typename> typename M>
const int MP_GMRES_IR_Solve<M>::DBL_PHASE = 2;

template <template <typename> typename M>
const int MP_GMRES_IR_Solve<M>::INIT_PHASE = MP_GMRES_IR_Solve<M>::HLF_PHASE;

template <template <typename> typename M>
class SimpleConstantThreshold : public MP_GMRES_IR_Solve<M>
{
protected:

    // *** PROTECTED CONSTANTS ***

    const double tol_hlf = pow(10, -02);
    const double tol_sgl = pow(10, -05);
    const double tol_dbl = pow(10, -10);

    // *** PROTECTED OVERRIDE METHODS ***

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
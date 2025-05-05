#ifndef VP_GMRES_IR_SOLVE_H
#define VP_GMRES_IR_SOLVE_H

#include "../IterativeRefinementBase.h"
#include "../../GMRES/GMRESSolve.h"

#include <cuda_fp16.h>

namespace cascade {

template <template <typename> typename TMatrix>
class VP_GMRES_IR_Solve: public IterativeRefinementBase<TMatrix>
{
private:

    TypedLinearSystem<TMatrix, double> * innerlinsys_dbl_ptr = nullptr;
    TypedLinearSystem<TMatrix, float> * innerlinsys_sgl_ptr = nullptr;
    TypedLinearSystem<TMatrix, __half> * innerlinsys_hlf_ptr = nullptr;

    TypedLinearSystem_MutAddlRHS<TMatrix, double> * mutrhs_innerlinsys_dbl_ptr = nullptr;
    TypedLinearSystem_MutAddlRHS<TMatrix, float> * mutrhs_innerlinsys_sgl_ptr = nullptr;
    TypedLinearSystem_MutAddlRHS<TMatrix, __half> * mutrhs_innerlinsys_hlf_ptr = nullptr;

    const PrecondArgPkg<TMatrix, double> orig_inner_precond_arg_pkg_dbl;
    PrecondArgPkg<TMatrix, float> * inner_precond_arg_pkg_sgl_ptr = nullptr;
    PrecondArgPkg<TMatrix, __half> * inner_precond_arg_pkg_hlf_ptr = nullptr;

    bool hlf_ptrs_instantiated() {
        return (
            (innerlinsys_hlf_ptr != nullptr) &&
            (mutrhs_innerlinsys_hlf_ptr != nullptr) &&
            (inner_precond_arg_pkg_hlf_ptr != nullptr)
        );
    };

    bool hlf_ptrs_empty() {
        return (
            (innerlinsys_hlf_ptr == nullptr) &&
            (mutrhs_innerlinsys_hlf_ptr == nullptr) &&
            (inner_precond_arg_pkg_hlf_ptr == nullptr)
        );
    };

    bool sgl_ptrs_instantiated() {
        return (
            (innerlinsys_sgl_ptr != nullptr) &&
            (mutrhs_innerlinsys_sgl_ptr != nullptr) &&
            (inner_precond_arg_pkg_sgl_ptr != nullptr)
        );
    };

    bool sgl_ptrs_empty() {
        return (
            (innerlinsys_sgl_ptr == nullptr) &&
            (mutrhs_innerlinsys_sgl_ptr == nullptr) &&
            (inner_precond_arg_pkg_sgl_ptr == nullptr)
        );
    };

    bool dbl_ptrs_instantiated() {
        return (
            (innerlinsys_dbl_ptr != nullptr) &&
            (mutrhs_innerlinsys_dbl_ptr != nullptr)
        );
    };

    bool dbl_ptrs_empty() {
        return (
            (innerlinsys_dbl_ptr == nullptr) &&
            (mutrhs_innerlinsys_dbl_ptr == nullptr)
        );
    };

    void delete_hlf_ptrs() {

        delete innerlinsys_hlf_ptr;
        innerlinsys_hlf_ptr = nullptr;

        delete mutrhs_innerlinsys_hlf_ptr;
        mutrhs_innerlinsys_hlf_ptr = nullptr;

        delete inner_precond_arg_pkg_hlf_ptr;
        inner_precond_arg_pkg_hlf_ptr = nullptr;

    }

    void delete_sgl_ptrs() {

        delete innerlinsys_sgl_ptr;
        innerlinsys_sgl_ptr = nullptr;

        delete mutrhs_innerlinsys_sgl_ptr;
        mutrhs_innerlinsys_sgl_ptr = nullptr;

        delete inner_precond_arg_pkg_sgl_ptr;
        inner_precond_arg_pkg_sgl_ptr = nullptr;

    }

    void delete_dbl_ptrs() {

        delete innerlinsys_dbl_ptr;
        innerlinsys_dbl_ptr = nullptr;

        delete mutrhs_innerlinsys_dbl_ptr;
        mutrhs_innerlinsys_dbl_ptr = nullptr;

    }

    void delete_ptrs() {
        delete_hlf_ptrs();
        delete_sgl_ptrs();
        delete_dbl_ptrs();
    }

    template <typename TPrecision>
    void setup_systems() {

        if (std::is_same<TPrecision, __half>::value) {

            if (hlf_ptrs_empty()) {
                innerlinsys_hlf_ptr = (
                    new TypedLinearSystem<TMatrix, __half>(
                        this->gen_lin_sys_ptr
                    )
                );
                mutrhs_innerlinsys_hlf_ptr = (
                    new TypedLinearSystem_MutAddlRHS<TMatrix, __half>(
                        innerlinsys_hlf_ptr, this->curr_res
                    )
                );
                inner_precond_arg_pkg_hlf_ptr = (
                    orig_inner_precond_arg_pkg_dbl.cast_hlf_ptr()
                );
            } else if (hlf_ptrs_instantiated()) {
                mutrhs_innerlinsys_hlf_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error(
                    "VP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<__half>"
                );
            }

        } else if (std::is_same<TPrecision, float>::value) {

            if (sgl_ptrs_empty()) {
                delete_hlf_ptrs();
                innerlinsys_sgl_ptr = (
                    new TypedLinearSystem<TMatrix, float>(this->gen_lin_sys_ptr)
                );
                mutrhs_innerlinsys_sgl_ptr = (
                    new TypedLinearSystem_MutAddlRHS<TMatrix, float>(
                        innerlinsys_sgl_ptr, this->curr_res
                    )
                );
                inner_precond_arg_pkg_sgl_ptr = (
                    orig_inner_precond_arg_pkg_dbl.cast_sgl_ptr()
                );
            } else if (sgl_ptrs_instantiated()) {
                mutrhs_innerlinsys_sgl_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error(
                    "VP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<float>"
                );
            }

        } else if (std::is_same<TPrecision, double>::value) {

            if (dbl_ptrs_empty()) {
                delete_sgl_ptrs();
                innerlinsys_dbl_ptr = (
                    new TypedLinearSystem<TMatrix, double>(
                        this->gen_lin_sys_ptr
                    )
                );
                mutrhs_innerlinsys_dbl_ptr = (
                    new TypedLinearSystem_MutAddlRHS<TMatrix, double>(
                        innerlinsys_dbl_ptr, this->curr_res
                    )
                );
            } else if (dbl_ptrs_instantiated()) {
                mutrhs_innerlinsys_dbl_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error(
                    "VP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<double>"
                );
            }

        } else {
            throw std::runtime_error(
                "VP_GMRES_IR_Solve: Invalid TPrecision used in "
                "setup_systems<TPrecision>"
            );
        }


    }

    template <typename TPrecision>
    void setup_inner_solve() {

        setup_systems<TPrecision>();

        if (std::is_same<TPrecision, __half>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<TMatrix, __half>>(
                mutrhs_innerlinsys_hlf_ptr,
                this->inner_solve_arg_pkg,
                *this->inner_precond_arg_pkg_hlf_ptr
            );

        } else if (std::is_same<TPrecision, float>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<TMatrix, float>>(
                mutrhs_innerlinsys_sgl_ptr,
                this->inner_solve_arg_pkg,
                *this->inner_precond_arg_pkg_sgl_ptr
            );

        } else if (std::is_same<TPrecision, double>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<TMatrix, double>>(
                mutrhs_innerlinsys_dbl_ptr,
                this->inner_solve_arg_pkg,
                orig_inner_precond_arg_pkg_dbl
            );

        } else {
            throw std::runtime_error(
                "VP_GMRES_IR_Solve: Invalid TPrecision used in "
                "setup_inner_solve<TPrecision>"
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
                "VP_GMRES_IR_Solve: Invalid cascade_phase in VP_GMRES_IR_Solver"
            );
        }

    }

protected:

    const double u_hlf = pow(2, -10);
    const double u_sgl = pow(2, -23);
    const double u_dbl = pow(2, -52);

    const static int HLF_PHASE = 0;
    const static int SGL_PHASE = 1;
    const static int DBL_PHASE = 2;
    const int INIT_PHASE;

    int cascade_phase;
    int hlf_sgl_cascade_change = -1;
    int sgl_dbl_cascade_change = -1;

    /* Determine which phase should be used based on current phase and current
       convergence progress */
    virtual int determine_next_phase() = 0;

    virtual void initialize_inner_outer_solver() override {
        // Initialize inner outer solver in lowest precision __half phase
        cascade_phase = INIT_PHASE;
        switch (cascade_phase) {
            case HLF_PHASE:
                setup_inner_solve<__half>(); break;
            case SGL_PHASE:
                setup_inner_solve<float>(); break;
            case DBL_PHASE:
                setup_inner_solve<double>(); break;
        }
    }

    virtual void outer_iterate_setup() override {
        // Specify inner_solver for outer_iterate_calc and setup
        int next_phase = determine_next_phase();
        if ((cascade_phase == HLF_PHASE) && (next_phase == SGL_PHASE)) {
            hlf_sgl_cascade_change = this->get_iteration();
        }
        if ((cascade_phase == SGL_PHASE) && (next_phase == DBL_PHASE)) {
            sgl_dbl_cascade_change = this->get_iteration();
        }
        cascade_phase = next_phase;
        choose_phase_solver();
    }

    virtual void derived_generic_reset() override {
        InnerOuterSolve<TMatrix>::derived_generic_reset();
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }
    
public:

    VP_GMRES_IR_Solve(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        ),
        int arg_init_phase = VP_GMRES_IR_Solve::HLF_PHASE
    ):
        IterativeRefinementBase<TMatrix>(arg_gen_lin_sys_ptr, arg_solve_arg_pkg),
        orig_inner_precond_arg_pkg_dbl(arg_inner_precond_arg_pkg_dbl),
        INIT_PHASE(arg_init_phase)
    {
        initialize_inner_outer_solver();
    }

    ~VP_GMRES_IR_Solve() {
        delete_ptrs();
    }

    // Forbid rvalue instantiation
    VP_GMRES_IR_Solve(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&,
        const PrecondArgPkg<TMatrix, double>
    ) = delete;

    int get_hlf_sgl_cascade_change() const { return hlf_sgl_cascade_change; }
    int get_sgl_dbl_cascade_change() const { return sgl_dbl_cascade_change; }

};

// Set solver to spend 20% in half phase and 40% in single phase and 40% time
// in double
template <template <typename> typename TMatrix>
class OuterRestartCount: public VP_GMRES_IR_Solve<TMatrix>
{
protected:

    const int hlf_iters;
    const int sgl_iters;

    virtual int determine_next_phase() override {

        if (this->cascade_phase == this->HLF_PHASE) {

            if (this->get_iteration() > hlf_iters) {
                return this->SGL_PHASE;
            } else {
                return this->cascade_phase;
            }

        } else if (this->cascade_phase == this->SGL_PHASE) {

            if (this->get_iteration() > hlf_iters+sgl_iters) {
                return this->DBL_PHASE;
            } else {
                return this->cascade_phase;
            }

        } else {
            return this->DBL_PHASE;
        }
    }

public:

    OuterRestartCount(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        ),
        int arg_init_phase = VP_GMRES_IR_Solve<TMatrix>::HLF_PHASE
    ):
        VP_GMRES_IR_Solve<TMatrix>(
            arg_gen_lin_sys_ptr,
            arg_solve_arg_pkg,
            arg_inner_precond_arg_pkg_dbl,
            arg_init_phase
        ),
        hlf_iters(arg_solve_arg_pkg.max_iter/5),
        sgl_iters(arg_solve_arg_pkg.max_iter*2/5)
    {}

};


// Set solver to 60% in single phase and 40% time in double
template <template <typename> typename TMatrix>
class SD_OuterRestartCount: public OuterRestartCount<TMatrix>
{
public:

    SD_OuterRestartCount(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        )
    ):
        OuterRestartCount<TMatrix>(
            arg_gen_lin_sys_ptr,
            arg_solve_arg_pkg,
            arg_inner_precond_arg_pkg_dbl,
            VP_GMRES_IR_Solve<TMatrix>::SGL_PHASE
        )
    {}

};


// Set solver to change phase when relative residual reaches an order of
// magnitude above roundoff
template <template <typename> typename TMatrix>
class RelativeResidualThreshold : public VP_GMRES_IR_Solve<TMatrix>
{
protected:

    const double tol_hlf = 10.*std::pow(2., -10);
    const double tol_sgl = 10.*std::pow(2., -23);

    virtual int determine_next_phase() override {

        if (this->cascade_phase == this->HLF_PHASE) {

            if (this->get_relres() <= tol_hlf) {
                return this->SGL_PHASE;
            } else {
                return this->cascade_phase;
            }

        } else if (this->cascade_phase == this->SGL_PHASE) {

            if (this->get_relres() <= tol_sgl) {
                return this->DBL_PHASE;
            } else {
                return this->cascade_phase;
            }

        } else {
            return this->DBL_PHASE;
        }
    }

public:

    using VP_GMRES_IR_Solve<TMatrix>::VP_GMRES_IR_Solve;

};

template <template <typename> typename TMatrix>
class SD_RelativeResidualThreshold: public RelativeResidualThreshold<TMatrix>
{
public:

    SD_RelativeResidualThreshold(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        )
    ):
    RelativeResidualThreshold<TMatrix>(
            arg_gen_lin_sys_ptr,
            arg_solve_arg_pkg,
            arg_inner_precond_arg_pkg_dbl,
            VP_GMRES_IR_Solve<TMatrix>::SGL_PHASE
        )
    {}

};

// Set solver to check for stagnation where the average change in elements
// is of order of being in a ball 2x times the size of roundoff
template <template <typename> typename TMatrix>
class CheckStagnation: public VP_GMRES_IR_Solve<TMatrix>
{
protected:

    virtual int determine_next_phase() override {

        int size = this->res_norm_history.size();
        double relative_progress;

        if (size < 2) {
            return this->INIT_PHASE;
        } else {
            relative_progress = -1.*(
                (this->res_norm_history[size-1] -
                 this->res_norm_history[size-2]) /
                this->res_norm_history[size-2]
            );
        }
        
        if (this->cascade_phase == this->HLF_PHASE) {
            if (relative_progress > 2.*this->u_hlf) {
                return this->cascade_phase;
            } else {
                return this->SGL_PHASE;
            }
        } else if (this->cascade_phase == this->SGL_PHASE) {
            if (relative_progress > 2.*this->u_sgl) {
                return this->cascade_phase;
            } else {
                return this->DBL_PHASE;
            }
        } else {
            return this->DBL_PHASE;
        }

    }

public:

    using VP_GMRES_IR_Solve<TMatrix>::VP_GMRES_IR_Solve;

};

template <template <typename> typename TMatrix>
class SD_CheckStagnation: public CheckStagnation<TMatrix>
{
public:

    SD_CheckStagnation(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        )
    ):
    CheckStagnation<TMatrix>(
            arg_gen_lin_sys_ptr,
            arg_solve_arg_pkg,
            arg_inner_precond_arg_pkg_dbl,
            VP_GMRES_IR_Solve<TMatrix>::SGL_PHASE
        )
    {}

};

// Set solver to use threshold for the first phase to partially mitigate risk
// of long time spend in low speed half cascade phase, then convert to
// stagnation check on the more bundled single phase to squeeze as much time
// out of that phase as possible
template <template <typename> typename TMatrix>
class ThresholdToStagnation: public VP_GMRES_IR_Solve<TMatrix>
{
protected:

    const double tol_hlf = 10.*std::pow(2., -10);

    virtual int determine_next_phase() override {
        
        if (this->cascade_phase == this->HLF_PHASE) {
    
            if (this->get_relres() <= tol_hlf) {
                return this->SGL_PHASE;
            } else {
                return this->cascade_phase;
            }
    
        } else if (this->cascade_phase == this->SGL_PHASE) {

            int size = this->res_norm_history.size();
            double relative_progress;
            if (size < 2) {
                return this->cascade_phase;
            } else {
                relative_progress = -1.*(
                    (this->res_norm_history[size-1] -
                    this->res_norm_history[size-2]) /
                    this->res_norm_history[size-2]
                );
            }

            if (relative_progress > 2.*this->u_sgl) {
                return this->cascade_phase;
            } else {
                return this->DBL_PHASE;
            }

        } else {
            return this->DBL_PHASE;
        }

    }

public:

    using VP_GMRES_IR_Solve<TMatrix>::VP_GMRES_IR_Solve;

};

}

#endif
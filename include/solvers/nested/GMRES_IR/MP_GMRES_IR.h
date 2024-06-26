#ifndef MP_GMRES_IR_SOLVE_H
#define MP_GMRES_IR_SOLVE_H

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

#include <cuda_fp16.h>

namespace cascade {

template <template <typename> typename TMatrix>
class MP_GMRES_IR_Solve: public IterativeRefinement<TMatrix>
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
                    "MP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<__half>"
                );
            }

        } else if (std::is_same<TPrecision, float>::value) {

            if (hlf_ptrs_instantiated() && sgl_ptrs_empty()) {
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
                    "MP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<float>"
                );
            }

        } else if (std::is_same<TPrecision, double>::value) {

            if (sgl_ptrs_instantiated() && dbl_ptrs_empty()) {
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
                    "MP_GMRES_IR_Solve: mismatching ptrs in "
                    "setup_systems<double>"
                );
            }

        } else {
            throw std::runtime_error(
                "MP_GMRES_IR_Solve: Invalid TPrecision used in "
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
                u_hlf,
                this->inner_solve_arg_pkg,
                *this->inner_precond_arg_pkg_hlf_ptr
            );

        } else if (std::is_same<TPrecision, float>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<TMatrix, float>>(
                mutrhs_innerlinsys_sgl_ptr,
                u_sgl,
                this->inner_solve_arg_pkg,
                *this->inner_precond_arg_pkg_sgl_ptr
            );

        } else if (std::is_same<TPrecision, double>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<TMatrix, double>>(
                mutrhs_innerlinsys_dbl_ptr,
                u_dbl,
                this->inner_solve_arg_pkg,
                orig_inner_precond_arg_pkg_dbl
            );

        } else {
            throw std::runtime_error(
                "MP_GMRES_IR_Solve: Invalid TPrecision used in "
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
                "MP_GMRES_IR_Solve: Invalid cascade_phase in MP_GMRES_IR_Solver"
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
    inline const static int INIT_PHASE = HLF_PHASE;

    int cascade_phase;
    int hlf_sgl_cascade_change = -1;
    int sgl_dbl_cascade_change = -1;

    /* Determine which phase should be used based on current phase and current
       convergence progress */
    virtual int determine_next_phase() = 0;

    void initialize_inner_outer_solver() override {
        // Initialize inner outer solver in lowest precision __half phase
        setup_inner_solve<__half>();
    }

    void outer_iterate_setup() override {
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

    void derived_generic_reset() override {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }
    
public:

    MP_GMRES_IR_Solve(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, double> arg_inner_precond_arg_pkg_dbl = (
            PrecondArgPkg<TMatrix, double>()
        )
    ):
        IterativeRefinement<TMatrix>(arg_gen_lin_sys_ptr, arg_solve_arg_pkg),
        orig_inner_precond_arg_pkg_dbl(arg_inner_precond_arg_pkg_dbl)
    {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }

    ~MP_GMRES_IR_Solve() {
        delete_ptrs();
    }

    // Forbid rvalue instantiation
    MP_GMRES_IR_Solve(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&,
        const PrecondArgPkg<TMatrix, double>
    ) = delete;

    int get_hlf_sgl_cascade_change() const { return hlf_sgl_cascade_change; }
    int get_sgl_dbl_cascade_change() const { return sgl_dbl_cascade_change; }

};

template <template <typename> typename TMatrix>
class SimpleConstantThreshold : public MP_GMRES_IR_Solve<TMatrix>
{
protected:

    const double tol_hlf = pow(10, -02);
    const double tol_sgl = pow(10, -05);
    const double tol_dbl = pow(10, -10);

    using MP_GMRES_IR_Solve<TMatrix>::MP_GMRES_IR_Solve;

    int determine_next_phase() override {
        if (this->cascade_phase == this->HLF_PHASE) {
            if ((this->get_relres() <= tol_hlf)) {
                return this->SGL_PHASE;
            } else {
                return this->cascade_phase;
            }
        } else if (this->cascade_phase == this->SGL_PHASE) {
            if ((this->get_relres() <= tol_sgl)) {
                return this->DBL_PHASE;
            } else {
                return this->cascade_phase;
            }
        } else {
            return this->DBL_PHASE;
        }
    }

};

template <template <typename> typename TMatrix>
class RestartCount: public MP_GMRES_IR_Solve<TMatrix>
{
protected:

    int hlf_iters = 4; // Set time spent in half iteration to min number
                       // of iter needed to save cost of cast as long as were
                       // guaranteed 1 MV product
    int sgl_iters = 2; // Set time spend in single iteration to min number
                       // of iter needed to save cost of cast as long as were
                       // guaranteed 1 MV product

    using MP_GMRES_IR_Solve<TMatrix>::MP_GMRES_IR_Solve;

    int determine_next_phase() override {
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

};

}

#endif
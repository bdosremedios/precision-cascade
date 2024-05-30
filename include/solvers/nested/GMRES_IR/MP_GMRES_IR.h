#ifndef MP_GMRES_IR_SOLVE_H
#define MP_GMRES_IR_SOLVE_H

#include <cuda_fp16.h>

#include "../IterativeRefinement.h"
#include "../../GMRES/GMRESSolve.h"

template <template <typename> typename M>
class MP_GMRES_IR_Solve: public IterativeRefinement<M>
{
private:

    TypedLinearSystem<M, double> * innerlinsys_dbl_ptr = nullptr;
    TypedLinearSystem<M, float> * innerlinsys_sgl_ptr = nullptr;
    TypedLinearSystem<M, __half> * innerlinsys_hlf_ptr = nullptr;

    TypedLinearSystem_MutAddlRHS<M, double> * mutrhs_innerlinsys_dbl_ptr = nullptr;
    TypedLinearSystem_MutAddlRHS<M, float> * mutrhs_innerlinsys_sgl_ptr = nullptr;
    TypedLinearSystem_MutAddlRHS<M, __half> * mutrhs_innerlinsys_hlf_ptr = nullptr;

    const PrecondArgPkg<M, double> &orig_inner_precond_arg_pkg_dbl;
    PrecondArgPkg<M, float> * inner_precond_arg_pkg_sgl_ptr = nullptr;
    PrecondArgPkg<M, __half> * inner_precond_arg_pkg_hlf_ptr = nullptr;

    // *** Helper Methods ***

    bool hlf_ptrs_instantiated() {
        return (
            (innerlinsys_hlf_ptr != nullptr) &&
            (mutrhs_innerlinsys_hlf_ptr != nullptr) //&&
            // (inner_precond_arg_pkg_hlf_ptr != nullptr)
        );
    };

    bool hlf_ptrs_empty() {
        return (
            (innerlinsys_hlf_ptr == nullptr) &&
            (mutrhs_innerlinsys_hlf_ptr == nullptr) //&&
            // (inner_precond_arg_pkg_hlf_ptr == nullptr)
        );
    };

    bool sgl_ptrs_instantiated() {
        return (
            (innerlinsys_sgl_ptr != nullptr) &&
            (mutrhs_innerlinsys_sgl_ptr != nullptr) //&&
            // (inner_precond_arg_pkg_sgl_ptr != nullptr)
        );
    };

    bool sgl_ptrs_empty() {
        return (
            (innerlinsys_sgl_ptr == nullptr) &&
            (mutrhs_innerlinsys_sgl_ptr == nullptr) //&&
            // (inner_precond_arg_pkg_sgl_ptr == nullptr)
        );
    };

    bool dbl_ptrs_instantiated() {
        return (innerlinsys_dbl_ptr != nullptr) && (mutrhs_innerlinsys_dbl_ptr != nullptr);
    };

    bool dbl_ptrs_empty() {
        return (innerlinsys_dbl_ptr == nullptr) && (mutrhs_innerlinsys_dbl_ptr == nullptr);
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

    // TODO: ADD SETUP FOR PRECONDITIONERS
    template <typename T>
    void setup_systems() {

        if (std::is_same<T, __half>::value) {

            if (hlf_ptrs_empty()) {
                innerlinsys_hlf_ptr = new TypedLinearSystem<M, __half>(this->gen_lin_sys_ptr);
                mutrhs_innerlinsys_hlf_ptr = new TypedLinearSystem_MutAddlRHS<M, __half>(
                    innerlinsys_hlf_ptr, this->curr_res
                );
            } else if (hlf_ptrs_instantiated()) {
                mutrhs_innerlinsys_hlf_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error("MP_GMRES_IR_Solve: mismatching ptrs in setup_inner_solve<__half>");
            }

        } else if (std::is_same<T, float>::value) {

            if (hlf_ptrs_instantiated() && sgl_ptrs_empty()) {
                delete_hlf_ptrs();
                innerlinsys_sgl_ptr = new TypedLinearSystem<M, float>(this->gen_lin_sys_ptr);
                mutrhs_innerlinsys_sgl_ptr = new TypedLinearSystem_MutAddlRHS<M, float>(
                    innerlinsys_sgl_ptr, this->curr_res
                );
            } else if (sgl_ptrs_instantiated()) {
                mutrhs_innerlinsys_sgl_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error("MP_GMRES_IR_Solve: mismatching ptrs in setup_inner_solve<float>");
            }

        } else if (std::is_same<T, double>::value) {

            if (sgl_ptrs_instantiated() && dbl_ptrs_empty()) {
                delete_sgl_ptrs();
                innerlinsys_dbl_ptr = new TypedLinearSystem<M, double>(this->gen_lin_sys_ptr);
                mutrhs_innerlinsys_dbl_ptr = new TypedLinearSystem_MutAddlRHS<M, double>(
                    innerlinsys_dbl_ptr, this->curr_res
                );
            } else if (dbl_ptrs_instantiated()) {
                mutrhs_innerlinsys_dbl_ptr->set_rhs(this->curr_res);
            } else {
                delete_ptrs();
                throw std::runtime_error("MP_GMRES_IR_Solve: mismatching ptrs in setup_inner_solve<double>");
            }

        } else {
            throw std::runtime_error("MP_GMRES_IR_Solve: Invalid T used in setup_systems<T>");
        }


    }

    template <typename T>
    void setup_inner_solve() {

        setup_systems<T>();

        if (std::is_same<T, __half>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<M, __half>>(
                mutrhs_innerlinsys_hlf_ptr, u_hlf, this->inner_solve_arg_pkg
            );

        } else if (std::is_same<T, float>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<M, float>>(
                mutrhs_innerlinsys_sgl_ptr, u_sgl, this->inner_solve_arg_pkg
            );

        } else if (std::is_same<T, double>::value) {

            this->inner_solver = std::make_shared<GMRESSolve<M, double>>(
                mutrhs_innerlinsys_dbl_ptr, u_dbl, this->inner_solve_arg_pkg
            );

        } else {
            throw std::runtime_error("MP_GMRES_IR_Solve: Invalid T used in setup_inner_solve<T>");
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
    int hlf_sgl_cascade_change = -1;
    int sgl_dbl_cascade_change = -1;

    // *** Virtual Abstract Methods ***

    // Determine which phase should be used based on current phase and current convergence progress
    virtual int determine_next_phase() = 0;

    // *** Concrete Override Methods ***

    // Initialize inner outer solver in __half phase
    void initialize_inner_outer_solver() override {
        setup_inner_solve<__half>();
    }

    // Specify inner_solver for outer_iterate_calc and setup
    void outer_iterate_setup() override {
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

    // *** Constructors ***
    MP_GMRES_IR_Solve(
        const GenericLinearSystem<M> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<M, double> &arg_inner_precond_arg_pkg_dbl = PrecondArgPkg<M, double>()
    ):
        IterativeRefinement<M>(arg_gen_lin_sys_ptr, arg_solve_arg_pkg),
        orig_inner_precond_arg_pkg_dbl(arg_inner_precond_arg_pkg_dbl)
    {
        cascade_phase = INIT_PHASE;
        initialize_inner_outer_solver();
    }

    ~MP_GMRES_IR_Solve() {
        delete_ptrs();
    }

    // Forbid rvalue instantiation
    MP_GMRES_IR_Solve(const GenericLinearSystem<M> * const, const SolveArgPkg &&) = delete;

    int get_hlf_sgl_cascade_change() const { return hlf_sgl_cascade_change; }
    int get_sgl_dbl_cascade_change() const { return sgl_dbl_cascade_change; }

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

    using MP_GMRES_IR_Solve<M>::MP_GMRES_IR_Solve;

};

template <template <typename> typename M>
class RestartCount: public MP_GMRES_IR_Solve<M>
{
protected:

    // *** Constants ***
    int hlf_iters = 4; // Set time spent in half iteration to min number of iter needed
                       // to save cost of cast as long as were guranteed 1 MV product
    int sgl_iters = 2; // Set time spend in single iteration to min number of iter needed
                       // to save cost of cast as long as were guranteed 1 MV product

    // *** Concrete Override Methods ***
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

    using MP_GMRES_IR_Solve<M>::MP_GMRES_IR_Solve;

};

#endif
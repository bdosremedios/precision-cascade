#ifndef ITERATIVE_SOLVE_H
#define ITERATIVE_SOLVE_H

#include "types/types.h"
#include "tools/arg_pkgs/argument_pkgs.h"
#include "preconditioners/implemented_preconditioners.h"

#include <vector>
#include <string>
#include <cmath>
#include <format>
#include <algorithm>
#include <iostream>
#include <iomanip>

template <template <typename> typename TMatrix>
class GenericIterativeSolve
{
private:

    void check_compatibility() const {
        if (gen_lin_sys_ptr->get_n() != init_guess.rows()) {
            throw std::runtime_error(
                "GenericIterativeSolve: initial guess not compatible"
            );
        }
        if (gen_lin_sys_ptr->get_m() != gen_lin_sys_ptr->get_n()) {
            throw std::runtime_error(
                "GenericIterativeSolve: A is not square"
            );
        }
    }

    void set_self_to_initial_state() {

        initiated = false;
        converged = false;
        terminated = false;
        curr_iter = 0;

        generic_soln = init_guess;

        initialize_instantiate_residual();

    }

    void initialize_instantiate_residual() {

        res_norm_history.clear();
        res_costheta_history.clear();
        curr_res = (
            gen_lin_sys_ptr->get_b() -
            gen_lin_sys_ptr->get_A()*init_guess
        );
        curr_res_norm = curr_res.norm();
        res_norm_history.push_back(curr_res_norm.get_scalar());
        res_costheta_history.push_back(0.);

    }

    void update_residual() {

        last_res = curr_res;
        curr_res = (
            gen_lin_sys_ptr->get_b() -
            gen_lin_sys_ptr->get_A()*generic_soln
        );
        curr_res_norm = curr_res.norm();
        res_norm_history.push_back(curr_res_norm.get_scalar());
        res_costheta_history.push_back(
            (last_res.dot(curr_res) /
            (last_res.norm()*curr_res_norm)).get_scalar()
        );

    }
    
protected:

    const GenericLinearSystem<TMatrix> * const gen_lin_sys_ptr;
    const Vector<double> init_guess;
    const double target_rel_res;

    int max_iter; // mutable to allow setting by derived solvers
    int curr_iter;
    bool initiated;
    bool converged;
    bool terminated;
    Vector<double> generic_soln = Vector<double>(cuHandleBundle());
    Vector<double> last_res = Vector<double>(cuHandleBundle());
    Vector<double> curr_res = Vector<double>(cuHandleBundle());
    Scalar<double> curr_res_norm;
    std::vector<double> res_norm_history;
    std::vector<double> res_costheta_history;

    GenericIterativeSolve(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ):
        gen_lin_sys_ptr(arg_gen_lin_sys_ptr),
        init_guess(
            arg_pkg.check_default_init_guess() ?
            make_guess(gen_lin_sys_ptr) : arg_pkg.init_guess
        ),
        max_iter(
            arg_pkg.check_default_max_iter() ?
            100 : arg_pkg.max_iter
        ),
        target_rel_res(
            arg_pkg.check_default_target_rel_res() ?
            1e-10 : arg_pkg.target_rel_res
        )
    {
        assert_valid_type<TMatrix>();
        check_compatibility();
        set_self_to_initial_state();
    }

    virtual void iterate() = 0;
    virtual void derived_generic_reset() = 0;

    Vector<double> make_guess(
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys_ptr
    ) const {
        return Vector<double>::Ones(
            arg_gen_lin_sys_ptr->get_cu_handles(),
            arg_gen_lin_sys_ptr->get_n()
        );
    }

    // Forbid rvalue instantiation
    GenericIterativeSolve(
        const GenericLinearSystem<TMatrix> * const,
        const SolveArgPkg &&
    ) = delete;

    // Disable copy constructor and copy assignment
    GenericIterativeSolve(const GenericIterativeSolve &) = delete;
    GenericIterativeSolve & operator=(GenericIterativeSolve &) = delete;

    /* Virtual destructor to determine destructors at runtime for correctness in
       dynamic memory usage*/
    virtual ~GenericIterativeSolve() = default;

public:

    Vector<double> get_generic_soln() const {
        return generic_soln;
    };
    Vector<double> get_curr_res() const {
        return curr_res;
    };
    double get_relres() const { 
        return curr_res_norm.get_scalar()/res_norm_history[0];
    }
    std::vector<double> get_res_norm_history() const {
        return res_norm_history;
    };
    std::vector<double> get_res_costheta_history() const {
        return res_costheta_history;
    };
    bool check_initiated() const { return initiated; };
    bool check_converged() const { return converged; };
    bool check_terminated() const { return terminated; };
    int get_iteration() const { return curr_iter; };

    // Perform solve with iterate() scheme updating generic_soln
    void solve() {

        if (initiated) {
            throw std::runtime_error(
                "Can not safely call solve without after a initiation without "
                "reset"
            );
        }
        initiated = true;
        // Set res_hist size here since max_iter is mutable before solve
        initialize_instantiate_residual();

        while (
            !converged &&
            ((curr_iter < max_iter) && (get_relres() > target_rel_res))
        ) {
            ++curr_iter;
            iterate();
            update_residual();
            if (terminated) { break; }
        }

        terminated = true;
        converged = (
            (get_relres() <= target_rel_res) ||
            ((curr_iter == 0) &&
             ((curr_res.norm() /
               gen_lin_sys_ptr->get_b().norm()).get_scalar() <= target_rel_res))
        );

    }

    void reset() {
        set_self_to_initial_state();
        derived_generic_reset();
    }

    std::string get_info_string() {
        return std::format(
            "Initiated: {} | Converged: {} | Current iter: {} | Current rel-res: {:.3g}",
            initiated,
            converged,
            curr_iter,
            get_relres()
        );
    }

    // Rudimentarily plot relative residual
    void view_relres_plot(std::string const &arg="normal") const {

        // Get max max_length entries to plot
        const int max_length(70);
        std::vector<double> plot_y;
        if (res_norm_history.size() > max_length) {
            double h = (res_norm_history.size()-1.0)/(max_length-1.0);
            for (int i=0; i<max_length; ++i) {
                plot_y.push_back(res_norm_history[static_cast<int>(i*h)]);
            }
        } else {
            plot_y = res_norm_history;
        }
        int length = plot_y.size();

        // Normalize to first residual norm
        double r_0 = plot_y[0];
        for (int i=0; i<length; ++i) { plot_y[i] /= r_0; }

        // If want log plot then take log of everything
        if (arg == "log") {
            for (int i=0; i<length; ++i) { plot_y[i] = log(plot_y[i])/log(10); }
        }

        // Find which in height buckets each plot point should be in
        const int height(12);
        std::vector<double> plot_y_bucket_index;
        std::vector<double> bucket_ends;
        double min_ = *std::min_element(plot_y.cbegin(), plot_y.cend());
        double max_ = *std::max_element(plot_y.cbegin(), plot_y.cend());

        // Get minimal of target relres and minimum if initiated for bottom of plot
        if (initiated) {
            if (arg == "log") {
                min_ = std::min(min_, log(target_rel_res)/log(10));
            } else {
                min_ = std::min(min_, target_rel_res);
            }
        }

        double bucket_width = (max_-min_)/static_cast<double>(height);
        for (double i=1; i<height; ++i) {
            bucket_ends.push_back(i*bucket_width + min_);
        }
        bucket_ends.push_back(max_);
        for (double y: plot_y) {
            for (int i=0; i<height; ++i) {
                if (y <= bucket_ends[i]) {
                    plot_y_bucket_index.push_back(i);
                    break;
                }
            }
        }

        // Iterate over grid and mark typed_soln wherever there is a plot entry
        const int min_length(3);
        std::cout << "Display of Relative Residual L-2 Norm: " << std::endl;
        if (arg == "log") {
            std::cout << std::setprecision(3) << std::pow(10, max_);
        } else {
            std::cout << std::setprecision(3) << max_;
        };
        std::cout << " " << std::string(std::max(min_length, length-1), '-')
                  << std::endl;
        for (int i=height-1; i>=0; --i) {
            for (int j=-1; j<length; ++j) {
                if ((j >= 0) && (plot_y_bucket_index[j] == i)) {
                    std::cout << "*";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }
        if (arg == "log") {
            std::cout << std::setprecision(3) << std::pow(10, min_);
        } else {
            std::cout << std::setprecision(3) << min_;
        };
        std::cout << " " << std::string(std::max(min_length, length-4), '-')
                  << std::endl;
        std::cout << "Iter: 0" << std::string(std::max(min_length, length-10), ' ')
                  << "Iter: " << curr_iter << std::endl;

    }

};

template <template <typename> typename TMatrix, typename TPrecision>
class TypedIterativeSolve: public GenericIterativeSolve<TMatrix>
{
private:

    void update_generic_soln() {
        this->generic_soln = typed_soln.template cast<double>();
    }
    
protected:

    const TypedLinearSystem_Intf<TMatrix, TPrecision> * const typed_lin_sys_ptr;
    const Vector<TPrecision> init_guess_typed;

    Vector<TPrecision> typed_soln = Vector<TPrecision>(cuHandleBundle());

    virtual void typed_iterate() = 0;
    virtual void derived_typed_reset() = 0;

    // Perform iteration updating typed_soln and then using that to
    // update generic_soln()
    void iterate() override {
        typed_iterate();
        update_generic_soln();
    }

    void derived_generic_reset() override {
        typed_soln = init_guess_typed;
        update_generic_soln();
        derived_typed_reset();
    }

public:

    TypedIterativeSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const arg_typed_lin_sys_ptr,
        const SolveArgPkg &arg_pkg
    ): 
        typed_lin_sys_ptr(arg_typed_lin_sys_ptr),
        init_guess_typed(
            (arg_pkg.check_default_init_guess()) ? 
                this->make_guess(
                    arg_typed_lin_sys_ptr->get_gen_lin_sys_ptr()
                ).template cast<TPrecision>() :
                arg_pkg.init_guess.template cast<TPrecision>()
        ),
        GenericIterativeSolve<TMatrix>(
            arg_typed_lin_sys_ptr->get_gen_lin_sys_ptr(),
            arg_pkg
        )
    {
        typed_soln = init_guess_typed;
        update_generic_soln();
    }

    // Forbid rvalue instantiation
    TypedIterativeSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const,
        const SolveArgPkg &&
    ) = delete;

    Vector<TPrecision> get_typed_soln() const { return typed_soln; };

};

#endif
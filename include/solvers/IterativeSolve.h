#ifndef ITERATIVE_SOLVE_H
#define ITERATIVE_SOLVE_H

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "types/types.h"
#include "tools/arg_pkgs/argument_pkgs.h"
#include "preconditioners/implemented_preconditioners.h"

template <template <typename> typename M>
class GenericIterativeSolve
{
private:

    void check_compatibility() const {
        if (lin_sys.get_n() != init_guess.rows()) {
            throw std::runtime_error("GenericIterativeSolve: initial guess not compatible");
        }
        if (lin_sys.get_m() != lin_sys.get_n()) {
            throw std::runtime_error("GenericIterativeSolve: A is not square");
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

        res_norm_hist.clear();
        res_hist = MatrixDense<double>(lin_sys.get_handle(), lin_sys.get_m(), max_iter+1);
        curr_res = lin_sys.get_b()-lin_sys.get_A()*init_guess;
        res_hist.get_col(0).set_from_vec(curr_res);
        curr_res_norm = curr_res.norm().get_scalar();
        res_norm_hist.push_back(curr_res_norm);

    }

    void update_residual() {

        curr_res = lin_sys.get_b()-lin_sys.get_A()*generic_soln;
        res_hist.get_col(curr_iter).set_from_vec(curr_res);
        curr_res_norm = curr_res.norm().get_scalar();
        res_norm_hist.push_back(curr_res_norm);

    }
    
protected:

    // *** Linear system attributes ***
    const GenericLinearSystem<M> &lin_sys;
    const Vector<double> init_guess;

    // *** Constant solve attributes ***
    const double target_rel_res;

    // *** Mutable solve attributes ***
    int max_iter; // mutable to allow setting by specific solvers
    int curr_iter;
    bool initiated;
    bool converged;
    bool terminated;
    Vector<double> generic_soln = Vector<double>(NULL);
    Vector<double> curr_res = Vector<double>(NULL);
    double curr_res_norm;
    MatrixDense<double> res_hist = MatrixDense<double>(NULL);
    std::vector<double> res_norm_hist;

    // *** Constructors ***
    GenericIterativeSolve(
        const GenericLinearSystem<M> &arg_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        lin_sys(arg_lin_sys),
        init_guess(
            arg_pkg.check_default_init_guess() ? make_guess(arg_lin_sys) : arg_pkg.init_guess
        ),
        max_iter(
            arg_pkg.check_default_max_iter() ? 100 : arg_pkg.max_iter
        ),
        target_rel_res(
            arg_pkg.check_default_target_rel_res() ? 1e-10 : arg_pkg.target_rel_res
        )
    {
        assert_valid_type<M>();
        check_compatibility();
        set_self_to_initial_state();
    }
    
    // *** Abstract virtual methods ***
    virtual void iterate() = 0;
    virtual void derived_generic_reset() = 0;

    static Vector<double> make_guess(const GenericLinearSystem<M> &arg_lin_sys) {
        return Vector<double>::Ones(arg_lin_sys.get_handle(), arg_lin_sys.get_n());
    }

    // Forbid rvalue instantiation
    GenericIterativeSolve(const GenericLinearSystem<M> &, const SolveArgPkg &&) = delete;
    GenericIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &) = delete;
    GenericIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &&) = delete;

    // Disable copy constructor and copy assignment
    GenericIterativeSolve(const GenericIterativeSolve &) = delete;
    GenericIterativeSolve & operator=(GenericIterativeSolve &) = delete;
    virtual ~GenericIterativeSolve() = default; // Virtual destructor to determine destructors at runtime for
                                                // correctness in dynamic memory usage

public:

    // *** Getters ***
    Vector<double> get_generic_soln() const { return generic_soln; };
    Vector<double> get_curr_res() const { return curr_res; };
    double get_relres() const { return curr_res_norm/res_norm_hist[0]; }
    MatrixDense<double> get_res_hist() const { return res_hist; };
    std::vector<double> get_res_norm_hist() const { return res_norm_hist; };
    bool check_initiated() const { return initiated; };
    bool check_converged() const { return converged; };
    bool check_terminated() const { return terminated; };
    int get_iteration() const { return curr_iter; };

    // Perform solve with iterate() scheme updating generic_soln
    void solve() {

        if (initiated) {
            throw std::runtime_error("Can not safely call solve without after a initiation without reset");
        }
        initiated = true;
        initialize_instantiate_residual(); // Set res_hist size here since max_iter is mutable before solve

        while (!converged && ((curr_iter < max_iter) && (get_relres() > target_rel_res))) {
            ++curr_iter;
            iterate();
            update_residual();
            if (terminated) { break; }
        }

        terminated = true;
        converged = (
            (get_relres() <= target_rel_res) ||
            (curr_iter == 0) && ((curr_res.norm()/lin_sys.get_b().norm()).get_scalar() <= target_rel_res)
        );

    }

    void reset() {
        set_self_to_initial_state();
        derived_generic_reset();
    }

    // Rudimentarily plot relative residual
    void view_relres_plot(std::string const &arg="normal") const {

        // Get max max_length entries to plot
        const int max_length(70);
        std::vector<double> plot_y;
        if (res_norm_hist.size() > max_length) {
            double h = (res_norm_hist.size()-1.0)/(max_length-1.0);
            for (int i=0; i<max_length; ++i) {
                plot_y.push_back(res_norm_hist[static_cast<int>(i*h)]);
            }
        } else {
            plot_y = res_norm_hist;
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
        for (double i=1; i<height; ++i) { bucket_ends.push_back(i*bucket_width+min_); }
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
        std::cout << " " << std::string(std::max(min_length, length-1), '-') << std::endl;
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
        std::cout << " " << std::string(std::max(min_length, length-4), '-') << std::endl;
        std::cout << "Iter: 0" << std::string(std::max(min_length, length-10), ' ')
                  << "Iter: " << curr_iter << std::endl;

    }

};

template <template <typename> typename M, typename T>
class TypedIterativeSolve: public GenericIterativeSolve<M>
{
private:

    void update_generic_soln() {
        this->generic_soln = typed_soln.template cast<double>();
    }
    
protected:

    // *** Const attributes ***
    const TypedLinearSystem<M, T> &typed_lin_sys;
    const Vector<T> init_guess_typed;

    // *** Mutable solve attributes ***
    Vector<T> typed_soln = Vector<T>(NULL);

    // *** Virtual abstract methods ***
    virtual void typed_iterate() = 0;
    virtual void derived_typed_reset() = 0;

    // *** Helper Methods ***

    // Perform iteration updating typed_soln and then using that to update generic_soln()
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

    // *** Constructors ***
    TypedIterativeSolve(
        const TypedLinearSystem<M, T> &arg_typed_lin_sys,
        const SolveArgPkg &arg_pkg
    ): 
        typed_lin_sys(arg_typed_lin_sys),
        init_guess_typed(
            (arg_pkg.check_default_init_guess()) ? 
                this->make_guess(arg_typed_lin_sys).template cast<T>() :
                arg_pkg.init_guess.template cast<T>()
        ),
        GenericIterativeSolve<M>(arg_typed_lin_sys, arg_pkg)
    {
        typed_soln = init_guess_typed;
        update_generic_soln();
    }

    // Forbid rvalue instantiation
    TypedIterativeSolve(const GenericLinearSystem<M> &, const SolveArgPkg &&) = delete;
    TypedIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &) = delete;
    TypedIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &&) = delete;

    // *** Getters ***
    Vector<T> get_typed_soln() const { return typed_soln; };

};

#endif
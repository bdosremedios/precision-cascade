#ifndef ITERATIVESOLVE_H
#define ITERATIVESOLVE_H

#include "types/types.h"
#include "tools/argument_pkgs.h"
#include "tools/Substitution.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

using std::shared_ptr, std::make_shared;
using std::vector;
using std::string;
using std::log, std::min, std::max, std::pow, std::sqrt;
using std::setprecision;
using std::cout, std::endl;

template <template <typename> typename M>
class GenericIterativeSolve
{
private:

    // *** PRIVATE HELPER METHODS ***

    void check_compatibility() const {

        // Ensure compatability to matrix and initial guess and squareness
        if (lin_sys.get_n() != init_guess.rows()) {
            throw runtime_error("A not compatible with initial guess init_guess");
        }
        if (lin_sys.get_m() != lin_sys.get_n()) {
            throw runtime_error("A is not square");
        }

    }

    void set_self_to_initial_state() {

        // Reset all variables
        initiated = false;
        converged = false;
        terminated = false;
        curr_iter = 0;
        res_norm_hist.clear();

        // Reset generic solution to initial guess
        generic_soln = init_guess;

        // Reset residual history to just initial residual
        res_hist = MatrixXd(lin_sys.get_m(), 1);
        curr_res = lin_sys.get_b()-lin_sys.get_A()*generic_soln;
        res_hist.col(0) = curr_res.to_matrix_dense();
        res_norm_hist.push_back(res_hist.col(0).norm());

    }
    
protected:

    // *** PROTECTED ATTRIBUTES ***

    // Linear system attributes
    const GenericLinearSystem<M> &lin_sys;
    const MatrixVector<double> init_guess;

    // Constant solve attributes
    const double target_rel_res;

    // Mutable solve Attributes
    int max_iter; // mutable to allow setting by specific solvers
    int curr_iter;
    bool initiated;
    bool converged;
    bool terminated;
    MatrixVector<double> generic_soln;
    MatrixVector<double> curr_res;
    MatrixDense<double> res_hist;
    vector<double> res_norm_hist;

    // *** PROTECTED CONSTRUCTORS ***
    GenericIterativeSolve(
        const GenericLinearSystem<M> &arg_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        lin_sys(arg_lin_sys),
        init_guess((arg_pkg.check_default_init_guess()) ? make_guess(arg_lin_sys) :
                                                          arg_pkg.init_guess),
        max_iter((arg_pkg.check_default_max_iter()) ? 100 :
                                                      arg_pkg.max_iter),
        target_rel_res((arg_pkg.check_default_target_rel_res()) ? 1e-10 :
                                                                  arg_pkg.target_rel_res)
    {
        assert_valid_type<M>();
        check_compatibility();
        set_self_to_initial_state();
    }

    // Forbid rvalue instantiation
    GenericIterativeSolve(const GenericLinearSystem<M> &, const SolveArgPkg &&) = delete;
    GenericIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &) = delete;
    GenericIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &&) = delete;
    
    // *** PROTECTED ABSTRACT METHODS ***
    
    // Perform update to generic_soln with iterative scheme
    virtual void iterate() = 0;

    // Perform reset specific to derived implemented class
    virtual void derived_generic_reset() = 0;

    // *** PROTECTED METHODS ***

    // Create initial guess based on system matrix arg_A
    MatrixVector<double> make_guess(const GenericLinearSystem<M> &arg_lin_sys) const {
        return MatrixVector<double>::Ones(arg_lin_sys.get_n());
    }

public:

    // *** PUBLIC METHODS ***

    // Getters
    MatrixVector<double> get_generic_soln() const { return generic_soln; };
    MatrixVector<double> get_curr_res() const { return curr_res; };
    double get_relres() const { return curr_res.norm()/res_norm_hist[0]; }
    MatrixDense<double> get_res_hist() const { return res_hist; };
    vector<double> get_res_norm_hist() const { return res_norm_hist; };
    bool check_initiated() const { return initiated; };
    bool check_converged() const { return converged; };
    bool check_terminated() const { return terminated; };
    int get_iteration() const { return curr_iter; };

    // Disable copy constructor and copy assignment
    GenericIterativeSolve(const GenericIterativeSolve &) = delete;
    GenericIterativeSolve & operator=(GenericIterativeSolve &) = delete;

    virtual ~GenericIterativeSolve() = default; // Virtual destructor to determine destructors at runtime for
                                                // correctness in dynamic memory usage

    // Perform solve with iterate() scheme updating generic_soln
    void solve() {

        // Mark as iterative solve started and expand res_hist to account for additional
        // residual information
        initiated = true;
        res_hist.conservativeResize(lin_sys.get_m(), max_iter+1); // Set res_hist size here since
                                                                  // max_iter is mutable before solve

        // Run while relative residual is still high, and under max iterations, and has not been
        // flagged as converged
        while(
            !converged &&
            ((curr_iter < max_iter) && (get_relres() > target_rel_res))
        ) {

            // Iterate solution
            ++curr_iter;

            iterate();

            // Update residual tracking
            curr_res = lin_sys.get_b()-lin_sys.get_A()*generic_soln;
            res_hist.col(curr_iter) = curr_res.to_matrix_dense();
            res_norm_hist.push_back(curr_res.norm());

            // Break early if terminated
            if (terminated) { break; }

        }

        // Ensure terminated if leave solve loop
        terminated = true;

        // On convergence flag as converged and remove extra zeros on x_hist.
        // Convergence is either a small relative residual or otherwise
        // if no iterations have been performed, that there is a small residual
        // relative to the RHS
        if (get_relres() <= target_rel_res) {
            res_hist.conservativeResize(lin_sys.get_m(), curr_iter+1);
            converged = true;
        } else if ((curr_iter == 0) && ((curr_res.norm()/lin_sys.get_b().norm()) <= target_rel_res)) {
            res_hist.conservativeResize(lin_sys.get_m(), curr_iter+1);
            converged = true;
        }

    } // end solve

    // Reset to initial state
    void reset() {
        set_self_to_initial_state();
        derived_generic_reset();
    }

    // Rudimentarily plot relative residual
    void view_relres_plot(string const &arg="normal") const {

        // Get max max_length entries to plot
        const int max_length(70);
        vector<double> plot_y;
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
        vector<double> plot_y_bucket_index;
        vector<double> bucket_ends;
        double min_ = *std::min_element(plot_y.cbegin(), plot_y.cend());
        double max_ = *std::max_element(plot_y.cbegin(), plot_y.cend());

        // Get minimal of target relres and minimum if initiated for bottom of plot
        if (initiated) {
            if (arg == "log") {
                min_ = min(min_, log(target_rel_res)/log(10));
            } else {
                min_ = min(min_, target_rel_res);
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
        cout << "Display of Relative Residual L-2 Norm: " << endl;
        if (arg == "log") {
            cout << setprecision(3) << pow(10, max_);
        } else {
            cout << setprecision(3) << max_;
        };
        cout << " " << string(max(min_length, length-1), '-') << endl;
        for (int i=height-1; i>=0; --i) {
            for (int j=-1; j<length; ++j) {
                if ((j >= 0) && (plot_y_bucket_index[j] == i)) {
                    cout << "*";
                } else {
                    cout << " ";
                }
            }
            cout << endl;
        }
        if (arg == "log") {
            cout << setprecision(3) << pow(10, min_);
        } else {
            cout << setprecision(3) << min_;
        };
        cout << " " << string(max(min_length, length-4), '-') << endl;
        cout << "Iter: 0" << string(max(min_length, length-10), ' ')
             << "Iter: " << curr_iter << endl;

    }

};

template <template <typename> typename M, typename T>
class TypedIterativeSolve: public GenericIterativeSolve<M>
{
private:

    // *** PRIVATE HELPER FUNCTIONS ***

    void update_generic_soln() {
        this->generic_soln = typed_soln.template cast<double>();
    }
    
protected:

    // *** PROTECTED ATTRIBUTES ***

    // Linear system attributes
    const TypedLinearSystem<M, T> &typed_lin_sys;

    // Constant solve attributes
    const MatrixVector<T> init_guess_typed;

    // Mutable solve attributes
    MatrixVector<T> typed_soln;

    // *** PROTECTED ABSTRACT METHODS ***
    
    // Perform update to typed_soln with iterative scheme
    virtual void typed_iterate() = 0;

    // Set abstract helper function for reset of derived function ensuring reset is considered
    // in derived implementation
    virtual void derived_typed_reset() = 0;

    // *** PROTECTED OVERRIDE METHODS ***

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

    // *** CONSTRUCTORS ***

    TypedIterativeSolve(
        const TypedLinearSystem<M, T> &arg_typed_lin_sys,
        const SolveArgPkg &arg_pkg
    ): 
        typed_lin_sys(arg_typed_lin_sys),
        init_guess_typed((arg_pkg.check_default_init_guess()) ?
            this->make_guess(arg_typed_lin_sys).template cast<T>() :
            arg_pkg.init_guess.template cast<T>()),
        GenericIterativeSolve<M>(arg_typed_lin_sys, arg_pkg)
    {
        typed_soln = init_guess_typed;
        update_generic_soln();
    }

    // Forbid rvalue instantiation
    TypedIterativeSolve(const GenericLinearSystem<M> &, const SolveArgPkg &&) = delete;
    TypedIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &) = delete;
    TypedIterativeSolve(const GenericLinearSystem<M> &&, const SolveArgPkg &&) = delete;

    // *** PUBLIC METHODS ***

    // Getters
    MatrixVector<T> get_typed_soln() const { return typed_soln; };

};

#endif
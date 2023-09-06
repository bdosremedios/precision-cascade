#ifndef ITERATIVESOLVE_H
#define ITERATIVESOLVE_H

#include "Eigen/Dense"

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

using Eigen::Matrix, Eigen::Dynamic;
using Eigen::placeholders::all;

using std::vector;
using std::cout, std::endl;

// Untyped abstract interface for untyped pointer access to typed interative solver interface
class GenericIterativeSolve {

    private:

        // *** PRIVATE HELPER METHODS ***

        void check_compatibility() const {

            // Ensure compatability to matrices, not empty, and square
            if ((m < 1) || (n < 1)) { throw runtime_error("Empty Matrix A"); }
            if (m != b.rows()) { throw runtime_error("A not compatible with b for linear system"); }
            if (n != init_guess.rows()) { throw runtime_error("A not compatible with initial guess init_guess"); }
            if (m != n) { throw runtime_error("A is not square"); }

        }

        void set_self_to_initial_state() {

            // Reset all variables
            initiated = false;
            converged = false;
            terminated = false;
            curr_outer_iter = 0;
            res_norm_hist.clear();

            // Reset generic solution to initial guess
            generic_soln = init_guess;

            // Reset residual history and set initial residual
            res_hist = Matrix<double, Dynamic, Dynamic>(m, max_outer_iter+1);
            res_hist(all, 0) = (b - A*generic_soln).template cast<double>();
            res_norm_hist.push_back(static_cast<double>(res_hist(all, 0).norm()));

        }
    
    protected:

        // *** PROTECTED ATTRIBUTES ***

        // Generic Linear System Attributes
        const int m; const int n;
        const Matrix<double, Dynamic, Dynamic> A;
        const Matrix<double, Dynamic, 1> b;
        const Matrix<double, Dynamic, 1> init_guess;

        // Generic Mutable Solve Attributes
        int max_outer_iter; // mutable to allow setting by specific solvers
        bool initiated;
        bool converged;
        bool terminated;
        int curr_outer_iter;
        Matrix<double, Dynamic, 1> generic_soln;

        // Constant solve attributes
        const double target_rel_res;

        // Mutable solve attributes
        Matrix<double, Dynamic, Dynamic> res_hist;
        vector<double> res_norm_hist;

        // *** PROTECTED CONSTRUCTORS ***
        GenericIterativeSolve(
            Matrix<double, Dynamic, Dynamic> arg_A,
            Matrix<double, Dynamic, 1> arg_b,
            Matrix<double, Dynamic, 1> arg_init_guess,
            int arg_max_outer_iter, double arg_target_rel_res
        ):
            A(arg_A), b(arg_b), init_guess(arg_init_guess),
            m(arg_A.rows()), n(arg_A.cols()),
            max_outer_iter(arg_max_outer_iter), target_rel_res(arg_target_rel_res)
        {
            check_compatibility();
            set_self_to_initial_state();
        }
        
        // *** PROTECTED ABSTRACT METHODS ***
        
        // Perform update to generic_soln with iterative scheme
        virtual void iterate() = 0;

        // Perform reset specific to derived implemented class
        virtual void derived_generic_reset() = 0;

    public:

        // *** METHODS **

        // Getters
        double get_relres() const { return res_norm_hist[curr_outer_iter]/res_norm_hist[0]; }
        Matrix<double, Dynamic, Dynamic> get_res_hist() const { return res_hist; };
        vector<double> get_res_norm_hist() const { return res_norm_hist; };
        bool check_initiated() const { return initiated; };
        bool check_converged() const { return converged; };
        bool check_terminated() const { return terminated; };
        int get_iteration() const { return curr_outer_iter; };

        // Disable copy constructor and copy assignment
        GenericIterativeSolve(GenericIterativeSolve const &) = delete;
        GenericIterativeSolve & operator=(GenericIterativeSolve &) = delete;

        virtual ~GenericIterativeSolve() = default; // Virtual to determine destructors at runtime for correctness
                                                  // in dynamic memory usage

        // Perform solve with iterate() scheme updating generic_soln
        void solve() {

            // Mark as iterative solve started
            initiated = true;

            // Run while relative residual is still high, and under max iterations, and has not been
            // flagged as converged
            double res_norm = res_norm_hist[0];
            while(!terminated &&
                 (!converged &&
                 ((curr_outer_iter < max_outer_iter) && ((res_norm/res_norm_hist[0]) > target_rel_res)))) {

                // Iterate solution
                ++curr_outer_iter;
                iterate();

                // Update residual tracking
                res_hist(all, curr_outer_iter) = b - A*generic_soln;
                res_norm = res_hist(all, curr_outer_iter).norm();
                res_norm_hist.push_back(res_norm);

            }

            // Ensure terminated if leave solve loop
            terminated = true;

            // On convergence flag as converged and remove extra zeros on x_hist.
            // Convergence is either a small relative residual or otherwise
            // if no iterations have been performed, that there is a small residual
            // relative to the RHS
            if ((res_norm/res_norm_hist[0]) <= target_rel_res) {
                res_hist.conservativeResize(m, curr_outer_iter+1);
                converged = true;
            } else if ((curr_outer_iter == 0) && ((res_norm/b.norm()) <= target_rel_res)) {
                res_hist.conservativeResize(m, curr_outer_iter+1);
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
                double h = (static_cast<double>(res_norm_hist.size())-1.0)/
                           (static_cast<double>(max_length)-1.0);
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
                for (int i=0; i<length; ++i) { plot_y[i] = std::log(plot_y[i])/std::log(10); }
            }

            // Find which in height buckets each plot point should be in
            const int height(12);
            vector<double> plot_y_bucket_index;
            vector<double> bucket_ends;
            double min = *std::min_element(plot_y.cbegin(), plot_y.cend());
            double max = *std::max_element(plot_y.cbegin(), plot_y.cend());
            
            // Get minimal of target relres and minimum if initiated for bottom of plot
            if (initiated) {
                if (arg == "log") {
                    min = std::min(min, std::log(target_rel_res)/std::log(10));
                } else {
                    min = std::min(min, target_rel_res);
                }
            }

            double bucket_width = (max-min)/static_cast<double>(height);
            for (double i=1; i<height; ++i) {bucket_ends.push_back(i*bucket_width+min);}
            bucket_ends.push_back(max);
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
                cout << std::setprecision(3) << std::pow(10, max);
            } else {
                cout << std::setprecision(3) << max;
            };
            cout << " " << string(std::max(min_length, length-1), '-') << endl;
            for (int i=height-1; i>=0; --i) {
                for (int j=-1; j<length; ++j) {
                    if (plot_y_bucket_index[j] == i) {
                        cout << "*";
                    } else {
                        cout << " ";
                    }
                }
                cout << endl;
            }
            if (arg == "log") {
                cout << std::setprecision(3) << std::pow(10, min);
            } else {
                cout << std::setprecision(3) << min;
            };
            cout << " " << string(std::max(min_length, length-4), '-') << endl;
            cout << "Iter: 1" << string(std::max(min_length, length-10), ' ')
                 << "Iter: " << curr_outer_iter << endl;

        } // end view_relres_plot

};

// Typed interface for implementation of shared but type dependent behavior
template <typename T>
class TypedIterativeSolve: public GenericIterativeSolve {

    private:

        // *** PRIVATE HELPER FUNCTIONS ***

        void update_generic_soln() {
            generic_soln = typed_soln.template cast<double>();
        }
    
    protected:

        // *** PROTECTED ATTRIBUTES ***

        // Typed Linear System Attributes
        const Matrix<T, Dynamic, Dynamic> A_T;
        const Matrix<T, Dynamic, 1> b_T;
        const Matrix<T, Dynamic, 1> init_guess_T;

        // Typed Mutable solve attributes
        Matrix<T, Dynamic, 1> typed_soln;

        // *** PROTECTED ABSTRACT METHODS ***
        
        // Perform update to typed_soln with iterative scheme
        virtual void typed_iterate() = 0;

        // Set abstract helper function for reset of derived function ensuring reset is considered
        // in derived implementation
        virtual void derived_typed_reset() = 0;

        // *** PROTECTED METHODS ***

        // Create initial guess based on system matrix arg_A
        Matrix<double, Dynamic, Dynamic> make_guess(Matrix<double, Dynamic, Dynamic> const &arg_A) const {
            return Matrix<double, Dynamic, Dynamic>::Ones(arg_A.cols(), 1);
        }

        // *** PROTECTED OVERRIDE METHODS ***

        // Perform iteration updating typed_soln and then using that to update generic_soln()
        void iterate() override {
            typed_iterate();
            update_generic_soln();
        }

        void derived_generic_reset() override {
            typed_soln = init_guess_T;
            update_generic_soln();
            derived_typed_reset();
        }

    public:

        // *** CONSTRUCTORS ***

        TypedIterativeSolve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ): 
            TypedIterativeSolve(
                arg_A, arg_b, make_guess(arg_A),
                arg_max_outer_iter, arg_target_rel_res
            )
        {}

        TypedIterativeSolve(
            Matrix<double, Dynamic, Dynamic> const &arg_A,
            Matrix<double, Dynamic, 1> const &arg_b, 
            Matrix<double, Dynamic, 1> const &arg_init_guess,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ): 
            A_T(arg_A.template cast<T>()),
            b_T(arg_b.template cast<T>()),
            init_guess_T(arg_init_guess.template cast<T>()),
            GenericIterativeSolve(
                arg_A, arg_b, arg_init_guess,
                arg_max_outer_iter, arg_target_rel_res
            )
        {
            typed_soln = init_guess_T;
            update_generic_soln();
        }

        // *** METHODS ***

        // Getters
        Matrix<T, Dynamic, 1> get_typed_soln() const { return typed_soln; };

};

#endif
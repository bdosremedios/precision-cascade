#ifndef ITERATIVESOLVE_H
#define ITERATIVESOLVE_H

#include "Eigen/Dense"
#include "Eigen/SparseCore"

#include "types/MatrixVector.h"
#include "types/MatrixDense.h"
#include "types/MatrixSparse.h"
#include "tools/ArgPkg.h"
#include "tools/Substitution.h"
#include "preconditioners/ImplementedPreconditioners.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Dynamic;

using std::shared_ptr, std::make_shared;
using std::vector;
using std::log, std::min, std::max, std::pow, std::sqrt;
using std::setprecision;
using std::cout, std::endl;

template <template <typename...> class, template<typename...> class> 
struct is_same_template : std::false_type{};

template <template <typename...> class T>
struct is_same_template<T,T> : std::true_type{};

// Generic abstract interface to iterative linear solve
template <template<typename> typename M>
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
            curr_iter = 0;
            res_norm_hist.clear();

            // Reset generic solution to initial guess
            generic_soln = init_guess;

            // Reset residual history to just initial residual
            res_hist = MatrixXd(m, 1);
            curr_res = b - A*generic_soln;
            res_hist.col(0) = curr_res;
            res_norm_hist.push_back(res_hist.col(0).norm());

        }
    
    protected:

        // *** PROTECTED ATTRIBUTES ***

        // Generic Linear System Attributes
        const int m; const int n;
        const M<double> A;
        const MatrixVector<double> b;
        const MatrixVector<double> init_guess;

        // Generic Mutable Solve Attributes
        int max_iter; // mutable to allow setting by specific solvers
        bool initiated;
        bool converged;
        bool terminated;
        int curr_iter;
        MatrixVector<double> generic_soln;
        MatrixVector<double> curr_res;

        // Constant solve attributes
        const double target_rel_res;

        // Mutable solve attributes, single element until solve is called
        MatrixXd res_hist;
        vector<double> res_norm_hist;

        // *** PROTECTED CONSTRUCTORS ***
        GenericIterativeSolve(
            M<double> const &arg_A, MatrixVector<double> const &arg_b, SolveArgPkg const &arg_pkg
        ):
            A(arg_A),
            b(arg_b),
            init_guess((arg_pkg.check_default_init_guess()) ? make_guess(arg_A) :
                                                              arg_pkg.init_guess),
            m(arg_A.rows()),
            n(arg_A.cols()),
            max_iter((arg_pkg.check_default_max_iter()) ? 100 : arg_pkg.max_iter),
            target_rel_res((arg_pkg.check_default_target_rel_res()) ? 1e-10 : arg_pkg.target_rel_res)
        {
            static_assert(
                ((is_same_template<M, MatrixSparse>::value) || (is_same_template<M, MatrixDense>::value)),
                "M argument must be type MatrixSparse or MatrixDense"
            );
            check_compatibility();
            set_self_to_initial_state();
        }
        
        // *** PROTECTED ABSTRACT METHODS ***
        
        // Perform update to generic_soln with iterative scheme
        virtual void iterate() = 0;

        // Perform reset specific to derived implemented class
        virtual void derived_generic_reset() = 0;

        // *** PROTECTED METHODS ***

        // Create initial guess based on system matrix arg_A
        MatrixVector<double> make_guess(M<double> const &arg_A) const {
            return MatrixVector<double>::Ones(arg_A.cols(), 1);
        }

    public:

        // *** PUBLIC METHODS **

        // Getters
        MatrixVector<double> get_generic_soln() const { return generic_soln; };
        MatrixVector<double> get_curr_res() const { return curr_res; };
        double get_relres() const { return res_norm_hist[curr_iter]/res_norm_hist[0]; }
        MatrixXd get_res_hist() const { return res_hist; };
        vector<double> get_res_norm_hist() const { return res_norm_hist; };
        bool check_initiated() const { return initiated; };
        bool check_converged() const { return converged; };
        bool check_terminated() const { return terminated; };
        int get_iteration() const { return curr_iter; };

        // Disable copy constructor and copy assignment
        GenericIterativeSolve(GenericIterativeSolve const &) = delete;
        GenericIterativeSolve & operator=(GenericIterativeSolve &) = delete;

        virtual ~GenericIterativeSolve() = default; // Virtual destructor to determine destructors at runtime for
                                                    // correctness in dynamic memory usage

        // Perform solve with iterate() scheme updating generic_soln
        void solve() {

            // Mark as iterative solve started and expand res_hist to account for additional
            // residual information
            initiated = true;
            res_hist.conservativeResize(m, max_iter+1); // Move here since max_iter is mutable
                                                        // before solve

            // Run while relative residual is still high, and under max iterations, and has not been
            // flagged as converged
            double res_norm = res_norm_hist[0];
            while(
                !converged && ((curr_iter < max_iter) && ((res_norm/res_norm_hist[0]) > target_rel_res))
            ) {

                // Iterate solution
                ++curr_iter;

                iterate();

                // Update residual tracking
                curr_res = b - A*generic_soln;
                res_hist.col(curr_iter) = curr_res;
                res_norm = res_hist.col(curr_iter).norm();
                res_norm_hist.push_back(res_norm);

                // Break early if terminated
                if (terminated) { break; }

            }

            // Ensure terminated if leave solve loop
            terminated = true;

            // On convergence flag as converged and remove extra zeros on x_hist.
            // Convergence is either a small relative residual or otherwise
            // if no iterations have been performed, that there is a small residual
            // relative to the RHS
            if ((res_norm/res_norm_hist[0]) <= target_rel_res) {
                res_hist.conservativeResize(m, curr_iter+1);
                converged = true;
            } else if ((curr_iter == 0) && ((res_norm/b.norm()) <= target_rel_res)) {
                res_hist.conservativeResize(m, curr_iter+1);
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
                    if (plot_y_bucket_index[j] == i) {
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

        } // end view_relres_plot

};

// Typed interface for implementation of shared but type dependent behavior
template <template<typename> typename M, typename T>
class TypedIterativeSolve: public GenericIterativeSolve<M> {

    private:

        // *** PRIVATE HELPER FUNCTIONS ***

        void update_generic_soln() {
            this->generic_soln = typed_soln.template cast<double>();
        }
    
    protected:

        // *** PROTECTED ATTRIBUTES ***

        // Typed Linear System Attributes
        const M<T> A_T;
        const MatrixVector<T> b_T;
        const MatrixVector<T> init_guess_T;

        // Typed Mutable solve attributes
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
            typed_soln = init_guess_T;
            update_generic_soln();
            derived_typed_reset();
        }

    public:

        // *** CONSTRUCTORS ***

        TypedIterativeSolve(
            M<double> const &arg_A, MatrixVector<double> const &arg_b, SolveArgPkg const &arg_pkg
        ): 
            A_T(arg_A.template cast<T>()),
            b_T(arg_b.template cast<T>()),
            init_guess_T((arg_pkg.check_default_init_guess()) ? this->make_guess(arg_A).template cast<T>() :
                                                                arg_pkg.init_guess.template cast<T>()),
            GenericIterativeSolve<M>(arg_A, arg_b, arg_pkg)
        {
            typed_soln = init_guess_T;
            update_generic_soln();
        }

        // *** METHODS ***

        // Getters
        MatrixVector<T> get_typed_soln() const { return typed_soln; };

};

#endif
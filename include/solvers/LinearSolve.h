#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

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

class GenericIterativeSolve {
    
}

template <typename T>
class LinearSolve {
    
    protected:

        // *** PROTECTED ATTRIBUTES ***

        // Linear System Attributes
        const Matrix<T, Dynamic, Dynamic> A;
        const Matrix<T, Dynamic, 1> b;
        const Matrix<T, Dynamic, 1> x_0;
        const int m; const int n;

        // Constant solve attributes
        const double target_rel_res;

        // Mutable solve attributes
        Matrix<T, Dynamic, 1> x;
        int max_outer_iter; // mutable to allow setting by specific solvers
        bool initiated = false;
        bool converged = false;
        bool terminated = false;
        int curr_outer_iter = 0;

        // Variables only valid once solve has been initiated
        Matrix<T, Dynamic, Dynamic> res_hist;
        vector<double> res_norm_hist;

        // *** PROTECTED ABSTRACT HELPER FUNCTIONS ***

        // Virtual function that returns advances the iterate x using previous iterates and
        // the linear solver's linear system
        // * WILL NOT BE CALLED IF CONVERGED = true so can assume converged is not true
        virtual void iterate() = 0;

        // Set abstract helper function for reset of derived function ensuring reset is considered
        // in derived implementation
        virtual void derived_reset() = 0;

        // *** PROTECTED HELPER FUNCTIONS ***

        // Create initial guess based on system matrix arg_A
        Matrix<T, Dynamic, Dynamic> make_guess(Matrix<T, Dynamic, Dynamic> const &arg_A) const {
            return Matrix<T, Dynamic, Dynamic>::Ones(arg_A.cols(), 1);
        }

    public:

        // *** CONSTRUCTORS ***

        LinearSolve(Matrix<T, Dynamic, Dynamic> const &arg_A,
                    Matrix<T, Dynamic, 1> const &arg_b,
                    int const &arg_max_outer_iter=100,
                    double const &arg_target_rel_res=1e-10
        ): 
            LinearSolve(
                arg_A, arg_b, make_guess(arg_A),
                arg_max_outer_iter, arg_target_rel_res
            )
        {}

        LinearSolve(Matrix<T, Dynamic, Dynamic> const &arg_A,
                    Matrix<T, Dynamic, 1> const &arg_b, 
                    Matrix<T, Dynamic, 1> const &arg_x_0,
                    int const &arg_max_outer_iter=100,
                    double const &arg_target_rel_res=1e-10
        ): 
            A(arg_A), b(arg_b), x_0(arg_x_0),
            max_outer_iter(arg_max_outer_iter), target_rel_res(arg_target_rel_res),
            m(arg_A.rows()), n(arg_A.cols())
        { constructorHelper(); }

        void constructorHelper() {

            // Ensure compatability to matrices, not empty, and square
            if ((m < 1) || (n < 1)) { throw runtime_error("Empty Matrix A"); }
            if (m != b.rows()) { throw runtime_error("A not compatible with b for linear system"); }
            if (n != x_0.rows()) { throw runtime_error("A not compatible with initial guess x_0"); }
            if (m != n) { throw runtime_error("A is not square"); }

            set_self_to_initial_state();

        };

        virtual ~LinearSolve() = default; // Virtual to determine destructors at runtime for correctness
                                          // in dynamic memory usage

        // Disable copy constructor and copy assignment
        LinearSolve(LinearSolve const &) = delete;
        LinearSolve & operator=(LinearSolve &) = delete;

        // *** METHODS ***

        // Getters
        Matrix<T, Dynamic, 1> get_soln() const { return x; };
        Matrix<T, Dynamic, Dynamic> get_res_hist() const { return res_hist; };
        vector<double> get_res_norm_hist() const { return res_norm_hist; };
        bool check_initiated() const { return initiated; };
        bool check_converged() const { return converged; };
        bool check_terminated() const { return terminated; };
        int get_iteration() const { return curr_outer_iter; };
        double get_relres() const { return res_norm_hist[curr_outer_iter]/res_norm_hist[0]; }

        // Reset LinearSolve to initial state
        void reset() {
            set_self_to_initial_state();
            derived_reset();
        }

        // Perform solve with LinearSolve scheme
        void solve() {

            // Mark as linear solve started
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
                res_hist(all, curr_outer_iter) = b - A*x;
                res_norm = res_hist(all, curr_outer_iter).norm();
                res_norm_hist.push_back(static_cast<double>(res_norm));

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

        // Rudimentarily plot relative residual
        void view_relres_plot(string const &arg="normal") const {

            // Get max max_length entries to plot
            int max_length = 70;
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
            for (int i=0; i<length; ++i) {
                plot_y[i] /= r_0;
            }

            // If want log plot then take log of everything
            if (arg == "log") {
                for (int i=0; i<length; ++i) {
                    plot_y[i] = std::log(plot_y[i])/std::log(10);
                }
            }

            // Find which in height buckets each plot point should be in
            int height = 12;
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

            // Iterate over grid and mark x wherever there is a plot entry
            cout << "Display of Relative Residual L-2 Norm: " << endl;
            if (arg == "log") {
                cout << std::setprecision(3) << std::pow(10, max);
            } else {
                cout << std::setprecision(3) << max;
            };
            cout << " " << string(min_three_int(length-1), '-') << endl;
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
            cout << " " << string(min_three_int(length-4), '-') << endl;
            cout << "Iter: 1" << string(min_three_int(length-10), ' ') << "Iter: " << curr_outer_iter << endl;

        }

    private:

        // *** PRIVATE HELPER FUNCTIONS ***

        void set_self_to_initial_state() {

            // Reset all variables
            initiated = false;
            converged = false;
            terminated = false;
            curr_outer_iter = 0;
            res_norm_hist.clear();

            // Set x as initial guess
            x = x_0;

            // Reset residual history and set initial residual
            res_hist = Matrix<T, Dynamic, Dynamic>(m, max_outer_iter+1);
            res_hist(all, 0) = b - A*x;
            res_norm_hist.push_back(static_cast<double>(res_hist(all, 0).norm()));

        }

        int min_three_int(int const &num) const {

            if (num < 3) {
                return 3;
            } else {
                return num;
            }

        }

};

template <typename T>
class LinearSolveTestingMock: public LinearSolve<T> {

    void iterate() override { x = soln; }
    void derived_reset() override {}

    public:
        Matrix<T, Dynamic, 1> soln;

        using LinearSolve<T>::A;
        using LinearSolve<T>::b;
        using LinearSolve<T>::x_0;
        using LinearSolve<T>::m;
        using LinearSolve<T>::n;
        using LinearSolve<T>::x;

        using LinearSolve<T>::initiated;
        using LinearSolve<T>::converged;
        using LinearSolve<T>::terminated;
        using LinearSolve<T>::curr_outer_iter;
        using LinearSolve<T>::max_outer_iter;
        using LinearSolve<T>::target_rel_res;
        using LinearSolve<T>::res_hist;
        using LinearSolve<T>::res_norm_hist;

        LinearSolveTestingMock(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_soln,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            soln(arg_soln),
            LinearSolve<T>::LinearSolve(
                arg_A, arg_b, arg_max_outer_iter, arg_target_rel_res
            )
        {}

        LinearSolveTestingMock(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            Matrix<T, Dynamic, 1> const &arg_soln,
            int const &arg_max_outer_iter=100,
            double const &arg_target_rel_res=1e-10
        ):
            soln(arg_soln),
            LinearSolve<T>::LinearSolve(
                arg_A, arg_b, arg_x_0, arg_max_outer_iter, arg_target_rel_res
            )
        {}

};

#endif
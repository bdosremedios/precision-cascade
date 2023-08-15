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

template <typename T>
class LinearSolve {
    
    protected:

        // Linear System Attributes
        Matrix<T, Dynamic, Dynamic> A;
        Matrix<T, Dynamic, 1> b;
        Matrix<T, Dynamic, 1> x_0;
        int m; int n;

        // Solution Tracking Attributes
        Matrix<T, Dynamic, 1> x;
        bool initiated = false;
        bool converged = false;
        bool terminated = false;
        int iteration = 0;

        // Variables only valid once solve has been initiated
        Matrix<T, Dynamic, Dynamic> res_hist;
        vector<double> res_norm_hist;

        // Virtual function that returns advances the iterate x using previous iterates and
        // the linear solver's linear system
        // * WILL NOT BE CALLED IF CONVERGED = true so can assume converged is not true
        virtual void iterate() = 0;

        // Plotting var
        double solve_target_relres;

    public:

        // Constructors/Destructors
        LinearSolve(Matrix<T, Dynamic, Dynamic> const &arg_A,
                    Matrix<T, Dynamic, 1> const &arg_b
        ) {
            constructorHelper(arg_A, arg_b, Matrix<T, Dynamic, 1>::Ones(arg_A.cols(), 1));
        }

        LinearSolve(Matrix<T, Dynamic, Dynamic> const &arg_A,
                    Matrix<T, Dynamic, 1> const &arg_b, 
                    Matrix<T, Dynamic, 1> const &arg_x_0
        ) {
            constructorHelper(arg_A, arg_b, arg_x_0);
        };

        void constructorHelper(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b, 
            Matrix<T, Dynamic, 1> const &arg_x_0
        ) {

                // Ensure compatability to matrices and not empty
                m = arg_A.rows();
                n = arg_A.cols();
                if ((m < 1) || (n < 1)) { throw runtime_error("Empty Matrix A"); }
                if (m != arg_b.rows()) { throw runtime_error("A not compatible with b for linear system"); }
                if (n != arg_x_0.rows()) { throw runtime_error("A not compatible with initial guess x_0"); }

                // Check matrix squareness
                if (this->m != this->n) { throw runtime_error("A is not square"); }

                // Load linear system variables if compatible
                A = arg_A;
                b = arg_b;
                x_0 = arg_x_0;

                // Load initial guess as initial solution
                x = x_0;

        };

        virtual ~LinearSolve() = default; // Virtual to determine destructors at runtime for correctness
                                          // in dynamic memory usage

        // Disable copy constructor and copy assignment
        LinearSolve(LinearSolve const &) = delete;
        LinearSolve & operator=(LinearSolve &) = delete;

        // Getters
        Matrix<T, Dynamic, 1> get_soln() { return x; };
        Matrix<T, Dynamic, Dynamic> get_res_hist() { return res_hist; };
        vector<double> get_res_norm_hist() { return res_norm_hist; };
        bool check_initiated() { return initiated; };
        bool check_converged() { return converged; };
        bool check_terminated() { return terminated; };
        int get_iteration() { return iteration; };
        double get_relres() {
            if (initiated) {
                return res_norm_hist[iteration]/res_norm_hist[0];
            } else {
                return -1;
            }
        }

        // Reset solve to initial state
        void reset() {

            // Reset all variables
            initiated = false;
            converged = false;
            terminated = false;
            iteration = 0;
            x = x_0;
            res_norm_hist.clear();

            // Leave res_hist since will be reset automatically on next solve
            Matrix<T, Dynamic, Dynamic> res_hist;

            // Call derived reset subroutine
            derived_reset();

        }

        // Set abstract function for reset of derived function ensuring reset is considered
        // in derived implementation
        virtual void derived_reset() = 0;

        // Perform linear solve with given iterate scheme
        void solve(int const &max_iter=100, double const &target_rel_res=1e-10) {

            // Store target residual
            solve_target_relres = target_rel_res;

            // Mark as linear solve started and start histories
            initiated = true;
            res_hist = Matrix<T, Dynamic, Dynamic>(m, max_iter+1);
            res_hist(all, 0) = b - A*x;
            double res_norm = res_hist(all, 0).norm();
            res_norm_hist.push_back(res_norm);

            // Run while relative residual is still high, and under max iterations, and has not been
            // flagged as converged
            while(!terminated && (!converged && ((iteration < max_iter) && ((res_norm/res_norm_hist[0]) > target_rel_res)))) {

                // Iterate solution
                ++iteration;
                iterate();

                // Update residual tracking
                res_hist(all, iteration) = b - A*x;
                res_norm = res_hist(all, iteration).norm();
                res_norm_hist.push_back(static_cast<double>(res_norm));

            }

            // Ensure terminated if leave solve loop
            terminated = true;

            // On convergence flag as converged and remove extra zeros on x_hist.
            // Convergence is either a small relative residual or otherwise
            // if no iterations have been performed, that there is a small residual
            // relative to the RHS
            if ((res_norm/res_norm_hist[0]) <= target_rel_res) {
                res_hist.conservativeResize(m, iteration+1);
                converged = true;
            } else if ((iteration == 0) && ((res_norm/b.norm()) <= target_rel_res)) {
                res_hist.conservativeResize(m, iteration+1);
                converged = true;
            }

        } // end solve

        // Helper for plotting
        int min_three_int(int const &num) const {
            if (num < 3) {
                return 3;
            } else {
                return num;
            }
        }

        // Plot relative residual
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
                    min = std::min(min, std::log(solve_target_relres)/std::log(10));
                } else {
                    min = std::min(min, solve_target_relres);
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
            cout << "Iter: 1" << string(min_three_int(length-10), ' ') << "Iter: " << iteration << endl;

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
        using LinearSolve<T>::iteration;
        using LinearSolve<T>::res_hist;
        using LinearSolve<T>::res_norm_hist;

        LinearSolveTestingMock(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_soln
        ): soln(arg_soln), LinearSolve<T>::LinearSolve(arg_A, arg_b) {}

        LinearSolveTestingMock(
            Matrix<T, Dynamic, Dynamic> const &arg_A,
            Matrix<T, Dynamic, 1> const &arg_b,
            Matrix<T, Dynamic, 1> const &arg_x_0,
            Matrix<T, Dynamic, 1> const &arg_soln
        ): soln(arg_soln), LinearSolve<T>::LinearSolve(arg_A, arg_b, arg_x_0) {}

};

#endif
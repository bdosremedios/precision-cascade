#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "Eigen/Dense"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

using Eigen::Matrix, Eigen::Dynamic;
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
        Matrix<T, Dynamic, Dynamic> x_hist;
        vector<double> res_norm_hist;

        // Virtual function that returns advances the iterate x using previous iterates and
        // the linear solver's linear system
        // * WILL NOT BE CALLED IF CONVERGED = true so can assume converged is not true
        virtual void iterate() = 0;

    public:

        // Constructors/Destructors
        LinearSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                    const Matrix<T, Dynamic, 1> arg_b) {
            constructorHelper(arg_A, arg_b, Matrix<T, Dynamic, 1>::Ones(arg_A.cols(), 1));
        }

        LinearSolve(const Matrix<T, Dynamic, Dynamic> arg_A,
                    const Matrix<T, Dynamic, 1> arg_b, 
                    const Matrix<T, Dynamic, 1> arg_x_0) {
            constructorHelper(arg_A, arg_b, arg_x_0);
        };

        void constructorHelper(
            const Matrix<T, Dynamic, Dynamic> arg_A,
            const Matrix<T, Dynamic, 1> arg_b, 
            const Matrix<T, Dynamic, 1> arg_x_0) {

                // Ensure compatability to matrices
                m = arg_A.rows();
                n = arg_A.cols();
                assert(((m == arg_b.rows()), "A not compatible with b for linear system"));
                assert(((n == arg_x_0.rows()), "A not compatible with initial guess x_0"));

                // Load linear system variables if compatible
                A = arg_A;
                b = arg_b;
                x_0 = arg_x_0;

                // Load initial guess as initial solution
                x = x_0;

        };

        virtual ~LinearSolve() = default; // Virtual to determine destructors at runtime for correctness
                                          // in dynamic memory usage

        // Getters
        Matrix<T, Dynamic, 1> soln() { return x; };
        Matrix<T, Dynamic, Dynamic> soln_hist() { return x_hist; };
        vector<double> get_res_norm_hist() { return res_norm_hist; };
        bool check_initiated() { return initiated; };
        bool check_converged() { return converged; };
        bool check_terminated() { return terminated; };
        int get_iteration() { return iteration; };
        double relres() {
            if (initiated) {
                return (b-A*x).norm()/res_norm_hist[0];
            } else {
                return -1;
            }
        }

        // Perform linear solve with given iterate scheme
        void solve(const int max_iter=100, const double target_rel_res=1e-10) {

            // Mark as linear solve started and start histories
            initiated = true;
            x_hist = Matrix<T, Dynamic, Dynamic>(m, max_iter+1);
            x_hist(Eigen::placeholders::all, 0) = x_0;
            double res_norm = (b - A*x).norm();
            res_norm_hist.push_back(res_norm);

            // Run while relative residual is still high, and under max iterations, and has not been
            // flagged as converged
            while(!terminated && (!converged && ((iteration < max_iter) && ((res_norm/res_norm_hist[0]) > target_rel_res)))) {

                // Iterate solution
                ++iteration;
                iterate();

                // Update accumulators
                res_norm = (b - A*x).norm();
                x_hist(Eigen::placeholders::all, iteration) = x;
                res_norm_hist.push_back(res_norm);

            }

            // On convergence flag as converged and remove extra zeros on x_hist.
            // Convergence is either a small relative residual or otherwise
            // if no iterations have been performed, that there is a small residual
            // relative to the RHS
            if ((res_norm/res_norm_hist[0]) <= target_rel_res) {
                x_hist.conservativeResize(m, iteration+1);
                converged = true;
            } else if ((iteration == 0) && ((res_norm/b.norm()) <= target_rel_res)) {
                x_hist.conservativeResize(m, iteration+1);
                converged = true;
            }

        } // end solve

        // Helper for plotting
        int min_three_int(int num) const {
            if (num < 3) {
                return 3;
            } else {
                return num;
            }
        }

        // Plot relative residual
        void view_relres_plot(string arg = "normal") const {

            // Get max max_length entries to plot
            int max_length = 70;
            vector<double> plot_y;
            if (res_norm_hist.size() > max_length) {
                double h = (static_cast<double>(res_norm_hist.size())-1.0)/(static_cast<double>(max_length)-1.0);
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
            cout << " " << string(min_three_int(length-1), '-') << endl;
            cout << "Iter: 1" << string(min_three_int(length-10), ' ') << "Iter: " << iteration << endl;

        }

};

#endif
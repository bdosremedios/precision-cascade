#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "Eigen/Dense"
#include <vector>
#include <algorithm>
#include <cmath>

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
        Matrix<T, Dynamic, Dynamic> x_hist;
        vector<double> res_norm_hist;
        bool initiated = false;
        bool converged = false;
        int iteration = 0;

        // Virtual function that returns the next iterate using previous iterates and
        // the linear solver's linear system
        virtual Matrix<T, Dynamic, 1> iterate() const = 0;

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

        };

        virtual ~LinearSolve() = default; // Virtual to determine destructors at runtime for correctness
                                          // in dynamic memory usage

        // Getters
        Matrix<T, Dynamic, 1> soln() { return x; };
        Matrix<T, Dynamic, Dynamic> soln_hist() { return x_hist; };
        vector<double> get_res_norm_hist() { return res_norm_hist; };
        bool check_converged() { return converged; };
        bool check_initiated() { return initiated; };

        // Perform linear solve with given iterate scheme
        void solve(const int max_iter=1000, const double target_rel_res=1e-12) {

            // Mark as linear solve started
            initiated = true;
            
            // Assume all vectors/matrices are compatible or else Eigen will return an error
            int m = A.rows();
            x = x_0;
            x_hist = Matrix<T, Dynamic, Dynamic>(m, max_iter+1);
            x_hist(Eigen::placeholders::all, 0) = x_0;
            double res_norm = (b - A*x).norm();
            res_norm_hist.push_back(res_norm);

            while(((res_norm/res_norm_hist[0]) > target_rel_res) && (iteration < max_iter)) {

                // Iterate solution and set new solution to it
                ++iteration;
                x = iterate();

                // Update accumulators
                res_norm = (b - A*x).norm();
                x_hist(Eigen::placeholders::all, iteration) = x;
                res_norm_hist.push_back(res_norm);

            }

            // On convergence flag as converged and remove extra zeros on x_hist
            if ((res_norm/res_norm_hist[0]) <= target_rel_res) {
                x_hist.conservativeResize(m, iteration+1);
                converged = true;
            }

        } // end solve

        void view_relres_plot(string arg = "normal") {

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
            if (arg != "log") {
                cout << "1 ";
            } else {
                cout << "10^0";
            }
            cout << string(length-1, '-') << endl;
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
            if (arg == "log") { cout << "10^"; };
            cout << static_cast<int>(min) << " " << string(length-1, '-') << endl;
            cout << "Iter: 1" << string(length-10, ' ') << "Iter: " << res_norm_hist.size() << endl;

        }

};

#endif
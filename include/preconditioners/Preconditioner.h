#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "Eigen/Dense"

using Eigen::Matrix, Eigen::Dynamic;

template <typename U>
class Preconditioner {

    public:

        Preconditioner() = default;
        virtual ~Preconditioner() = default;

        // Abstract function to apply action of inverse M on given vector
        virtual Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const = 0;

        // Abstract functions to check compatibility of preconditioner with linear system
        // on both left and right
        virtual bool check_compatibility_left(int m) const = 0;
        virtual bool check_compatibility_right(int n) const = 0;

};

#endif
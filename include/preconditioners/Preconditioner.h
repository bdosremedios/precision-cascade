#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "Eigen/Dense"

using Eigen::Matrix, Eigen::Dynamic;

template <typename U>
class Preconditioner {

    public:

        Preconditioner() = default;
        virtual ~Preconditioner() = default;

        // Apply action of inverse M on given vector
        virtual Matrix<U, Dynamic, 1> action_inv_M(Matrix<U, Dynamic, 1> vec) const = 0;

};

#endif
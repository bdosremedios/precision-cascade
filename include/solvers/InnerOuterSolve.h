#ifndef INNEROUTERSOLVE_H
#define INNEROUTERSOLVE_H

#include "IterativeSolve.h"

#include "Eigen/Dense"

#include <iostream>
#include <memory>

using std::shared_ptr;

class InnerOuterSolve: public GenericIterativeSolve {

    private:

        shared_ptr<GenericIterativeSolve> inner_solver;

        void outer_iterate();

    public:

        void solve() override {
            outer_iterate();
        }

};

#endif
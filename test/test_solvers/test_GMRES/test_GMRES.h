#ifndef TEST_GMRES_H
#define TEST_GMRES_H

#include "solvers/krylov/GMRES.h"

template <typename T, typename U=T>
class GMRESSolveTestingMock: public GMRESSolve<T, U> {

    public:

        using GMRESSolve<T, U>::GMRESSolve;
        using GMRESSolve<T, U>::typed_soln;

        using GMRESSolve<T, U>::H;
        using GMRESSolve<T, U>::Q_kry_basis;
        using GMRESSolve<T, U>::Q_H;
        using GMRESSolve<T, U>::R_H;

        using GMRESSolve<T, U>::kry_space_dim;
        using GMRESSolve<T, U>::max_kry_space_dim;
        using GMRESSolve<T, U>::next_q;
        using GMRESSolve<T, U>::rho;
        using GMRESSolve<T, U>::max_iter;

        using GMRESSolve<T, U>::update_QR_fact;
        using GMRESSolve<T, U>::update_x_minimizing_res;
        using GMRESSolve<T, U>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
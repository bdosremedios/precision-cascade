#ifndef TEST_GMRES_H
#define TEST_GMRES_H

#include "../../test.h"

#include "solvers/GMRES/GMRES.h"

template <template <typename> typename M, typename T, typename W=T>
class GMRESSolveTestingMock: public GMRESSolve<M, T, W> {

    public:

        using GMRESSolve<M, T, W>::GMRESSolve;
        using GMRESSolve<M, T, W>::typed_soln;

        using GMRESSolve<M, T, W>::H;
        using GMRESSolve<M, T, W>::Q_kry_basis;
        using GMRESSolve<M, T, W>::Q_H;
        using GMRESSolve<M, T, W>::R_H;

        using GMRESSolve<M, T, W>::kry_space_dim;
        using GMRESSolve<M, T, W>::max_kry_space_dim;
        using GMRESSolve<M, T, W>::next_q;
        using GMRESSolve<M, T, W>::rho;
        using GMRESSolve<M, T, W>::max_iter;

        using GMRESSolve<M, T, W>::update_QR_fact;
        using GMRESSolve<M, T, W>::update_x_minimizing_res;
        using GMRESSolve<M, T, W>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
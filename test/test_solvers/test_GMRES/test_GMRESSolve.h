#ifndef TEST_GMRES_H
#define TEST_GMRES_H

#include "../../test.h"

#include "solvers/GMRES/GMRESSolve.h"

template <template <typename> typename M, typename T>
class GMRESSolveTestingMock: public GMRESSolve<M, T> {

    public:

        using GMRESSolve<M, T>::GMRESSolve;
        using GMRESSolve<M, T>::typed_soln;

        using GMRESSolve<M, T>::Q_kry_basis;
        using GMRESSolve<M, T>::H_k;
        using GMRESSolve<M, T>::H_Q;
        using GMRESSolve<M, T>::H_R;

        using GMRESSolve<M, T>::curr_kry_dim;
        using GMRESSolve<M, T>::max_kry_dim;
        using GMRESSolve<M, T>::next_q;
        using GMRESSolve<M, T>::rho;
        using GMRESSolve<M, T>::max_iter;

        using GMRESSolve<M, T>::update_QR_fact;
        using GMRESSolve<M, T>::update_x_minimizing_res;
        using GMRESSolve<M, T>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
#ifndef TEST_GMRES_H
#define TEST_GMRES_H

#include "test.h"

#include "solvers/GMRES/GMRESSolve.h"

template <template <typename> typename TMatrix, typename TPrecision>
class GMRESSolveTestingMock: public GMRESSolve<TMatrix, TPrecision> {

    public:

        using GMRESSolve<TMatrix, TPrecision>::GMRESSolve;
        using GMRESSolve<TMatrix, TPrecision>::typed_soln;

        using GMRESSolve<TMatrix, TPrecision>::Q_kry_basis;
        using GMRESSolve<TMatrix, TPrecision>::H_k;
        using GMRESSolve<TMatrix, TPrecision>::H_Q;
        using GMRESSolve<TMatrix, TPrecision>::H_R;

        using GMRESSolve<TMatrix, TPrecision>::curr_kry_dim;
        using GMRESSolve<TMatrix, TPrecision>::max_kry_dim;
        using GMRESSolve<TMatrix, TPrecision>::next_q;
        using GMRESSolve<TMatrix, TPrecision>::rho;
        using GMRESSolve<TMatrix, TPrecision>::max_iter;

        using GMRESSolve<TMatrix, TPrecision>::update_QR_fact;
        using GMRESSolve<TMatrix, TPrecision>::update_x_minimizing_res;
        using GMRESSolve<TMatrix, TPrecision>::iterate;

        void iterate_no_soln_solve() {
            this->update_subspace_k();
            this->update_nextq_and_Hkplus1();
            this->check_termination();
        }

};

#endif
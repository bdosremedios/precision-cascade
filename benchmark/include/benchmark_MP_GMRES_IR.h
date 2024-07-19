#ifndef BENCHMARK_MP_GMRES_IR_H
#define BENCHMARK_MP_GMRES_IR_H

#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Benchmark_MP_GMRES_IR: public Benchmark_Nested_GMRES
{
public:

    template <template <template <typename> typename> typename TSolver>
    void execute_mp_gmres_ir(
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        SolveArgPkg args(nested_gmres_outer_iters, nested_gmres_inner_iters, 0.);

        clock.clock_start();
        TSolver<NoFillMatrixSparse> mp_restarted_gmres(
            &gen_lin_sys, args
        );
        mp_restarted_gmres.solve();
        clock.clock_stop();

    };

};

#endif
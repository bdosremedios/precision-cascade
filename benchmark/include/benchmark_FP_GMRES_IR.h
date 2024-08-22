#ifndef BENCHMARK_FP_GMRES_IR_H
#define BENCHMARK_FP_GMRES_IR_H

#include "benchmark_Nested_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

template <template <typename> typename TMatrix, typename TPrecision>
class NoProgress_FP_GMRES_IR: public cascade::FP_GMRES_IR_Solve<TMatrix, TPrecision>
{
public:

    using cascade::FP_GMRES_IR_Solve<TMatrix, TPrecision>::FP_GMRES_IR_Solve;

    // Remove any progress towards solution
    void outer_iterate_complete() override {
        this->generic_soln = this->init_guess;
    }

};

class Benchmark_FP_GMRES_IR: public Benchmark_Nested_GMRES
{
public:

    template <typename TPrecision>
    void execute_fp_gmres_ir(
        Benchmark_AccumClock &clock, NoFillMatrixSparse<double> &A
    ) {

        Vector<double> x_soln = Vector<double>::Random(
            BenchmarkBase::bundle, A.rows()
        );

        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(A, A*x_soln);
        TypedLinearSystem<NoFillMatrixSparse, TPrecision> typed_lin_sys(
            &gen_lin_sys
        );
        SolveArgPkg args(nested_gmres_outer_iters, nested_gmres_inner_iters, 0.);

        clock.clock_start();
        NoProgress_FP_GMRES_IR<NoFillMatrixSparse, TPrecision> fp_restarted_gmres(
            &typed_lin_sys, 0., args
        );
        fp_restarted_gmres.solve();
        clock.clock_stop();
        std::cout << fp_restarted_gmres.get_info_string() << std::endl;
        std::cout << "Inner iterations: [";
        for (int i=0; i<fp_restarted_gmres.get_inner_iterations().size()-1; ++i) {
            std::cout << fp_restarted_gmres.get_inner_iterations()[i] << " ";
        }
        std::cout << fp_restarted_gmres.get_inner_iterations()[
                        fp_restarted_gmres.get_inner_iterations().size()-1
                     ]
                  << "]" << std::endl;
        // std::cout << "Res norm history: [";
        // for (int i=0; i<fp_restarted_gmres.get_res_norm_history().size()-1; ++i) {
        //     std::cout << fp_restarted_gmres.get_res_norm_history()[i] << " ";
        // }
        // std::cout << fp_restarted_gmres.get_res_norm_history()[
        //                 fp_restarted_gmres.get_res_norm_history().size()-1
        //              ]
        //           << "]" << std::endl;

    };

};

#endif
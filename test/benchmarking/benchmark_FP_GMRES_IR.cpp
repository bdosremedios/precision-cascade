#include "include/benchmark_Sparse_Nested.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

class Benchmark_FP_GMRES_IR_Sparse: public Benchmark_Sparse_Nested {};

TEST_F(Benchmark_FP_GMRES_IR_Sparse, FP_GMRES_IR_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [this] (
        NoFillMatrixSparse<double> &A, Vector<double> &x_temp
    ) {

        Vector<double> b = A*x_temp;

        TypedLinearSystem<NoFillMatrixSparse, double> lin_sys(A, b);
        SolveArgPkg args(nested_outer_iter, nested_inner_iter, 0.);

        FP_GMRES_IR_Solve fp_restarted_gmres(lin_sys, 0., args);

        fp_restarted_gmres.solve();
        std::cout << fp_restarted_gmres.get_info_string() << std::endl;

    };

    basic_func_benchmark<NoFillMatrixSparse>(
        nested_n_min, nested_n_max, make_norm_A, execute_func, "fp_gmres_ir"
    );

}
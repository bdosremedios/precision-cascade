#include "include/benchmark_Sparse.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"

class Benchmark_GMRESSolve_Sparse: public Benchmark_Sparse
{
public:

    int gmres_n_min = 11;
    int gmres_n_max = 17-prototying_n_speed_up;
    
    double gmressolve_iters = 50;

};

TEST_F(Benchmark_GMRESSolve_Sparse, GMRESSolve_BENCHMARK) {

    std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [this] (
        NoFillMatrixSparse<double> &A, Vector<double> &x_temp
    ) {

        Vector<double> b = A*x_temp;

        TypedLinearSystem<NoFillMatrixSparse, double> lin_sys(A, b);
        SolveArgPkg args(gmressolve_iters, SolveArgPkg::default_max_inner_iter, 0);

        GMRESSolve gmres(lin_sys, 0., args);

        gmres.solve();
        std::cout << gmres.get_info_string() << std::endl;

    };

    basic_func_benchmark<NoFillMatrixSparse>(
        gmres_n_min, gmres_n_max, make_A, execute_func, "gmressolve"
    );

}
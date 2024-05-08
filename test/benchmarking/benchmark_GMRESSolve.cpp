#include "include/benchmark_Sparse.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"

class Benchmark_GMRESSolve_Sparse: public Benchmark_Sparse
{
public:

    int gmres_n_min = 8;
    int gmres_n_max = 16-prototying_n_speed_up;
    
    double gmressolve_iters = 100;

    double gmresolve_col_non_zeros = 1000.;

    std::function<NoFillMatrixSparse<double> (int, int)> make_A = [this] (
        int m, int n
    ) -> NoFillMatrixSparse<double> {
        return NoFillMatrixSparse<double>::Random(
            TestBase::bundle, m, m,
            (gmresolve_col_non_zeros/static_cast<double>(m) > 1) ? 1 : gmresolve_col_non_zeros/static_cast<double>(m)
        );
    };

};

// TEST_F(Benchmark_GMRESSolve_Sparse, GMRESSolve_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [this] (
//         NoFillMatrixSparse<double> &A, Vector<double> &x_temp
//     ) {

//         Vector<double> b = A*x_temp;

//         TypedLinearSystem<NoFillMatrixSparse, double> lin_sys(A, b);
//         SolveArgPkg args(gmressolve_iters, SolveArgPkg::default_max_inner_iter, 0);

//         GMRESSolve gmres(lin_sys, 0., args);

//         gmres.solve();

//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         gmres_n_min, gmres_n_max, make_A, execute_func, "gmressolve"
//     );

// }

TEST_F(Benchmark_GMRESSolve_Sparse, GMRESSolve_Figure_out_15_BENCHMARK) {

    int m = std::pow(2, 15);
    Vector<double> b = Vector<double>::Random(TestBase::bundle, m);
    TypedLinearSystem<NoFillMatrixSparse, double> lin_sys(make_A(m, m), b);
    std::cout << lin_sys.get_A().get_info_string() << std::endl;

    SolveArgPkg args(gmressolve_iters, SolveArgPkg::default_max_inner_iter, 0);
    GMRESSolve gmres(lin_sys, 0., args);

}
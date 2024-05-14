#include "include/benchmark_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"

class Playground: public TestBase {};

TEST_F(Playground, Playground_BENCHMARK) {
    
    int m(10000);
    TypedLinearSystem<NoFillMatrixSparse, double> lin_sys(
        NoFillMatrixSparse<double>::Random(TestBase::bundle, m, m, 0.1),
        Vector<double>::Random(TestBase::bundle, m)
    );
    SolveArgPkg args(20, SolveArgPkg::default_max_inner_iter, 1e-10);

    GMRESSolve gmres(lin_sys, 0., args);

    gmres.solve();

    std::cout << gmres.get_info_string() << std::endl;

}
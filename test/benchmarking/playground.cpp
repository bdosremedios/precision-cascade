#include "include/benchmark_GMRES.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/GMRES/GMRESSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Playground: public TestBase {};

TEST_F(Playground, Playground_BENCHMARK) {
    
    int m(50000);

    for (int i=0; i<5; ++i) {
        GenericLinearSystem<NoFillMatrixSparse> gen_lin_sys(
            NoFillMatrixSparse<double>::Random(TestBase::bundle, m, m, 200./static_cast<double>(m)),
            Vector<double>::Random(TestBase::bundle, m)
        );
        TypedLinearSystem<NoFillMatrixSparse, double> typed_lin_sys(&gen_lin_sys);
        SolveArgPkg gmres_args(30, 30, 1e-10);
        SolveArgPkg args(5, 30, 1e-10);

        GMRESSolve gmres(&typed_lin_sys, 0., gmres_args);

        gmres.solve();

        std::cout << gmres.get_info_string() << std::endl;
    
        // RestartCount<NoFillMatrixSparse> mp_restarted_gmres(&gen_lin_sys, args);
        // mp_restarted_gmres.solve();

        // std::cout << mp_restarted_gmres.get_info_string() << std::endl;
    }

}
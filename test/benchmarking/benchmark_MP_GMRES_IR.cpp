#include "include/benchmark_Sparse_Nested.h"

#include "tools/arg_pkgs/LinearSystem.h"
#include "tools/arg_pkgs/SolveArgPkg.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

class Benchmark_MP_GMRES_IR_Sparse: public Benchmark_Sparse_Nested {};

TEST_F(Benchmark_MP_GMRES_IR_Sparse, MP_GMRES_IR_BENCHMARK) {

    int m = 1500;
    NoFillMatrixSparse<double> mat = make_norm_A(m, m);
    std::cout << mat.get_info_string() << std::endl;

    NoFillMatrixSparse<float> mat_sgl = mat.cast<float>();
    std::cout << mat_sgl.get_info_string() << std::endl;

    NoFillMatrixSparse<__half> mat_hlf = mat.cast<__half>();
    std::cout << mat_hlf.get_info_string() << std::endl;

    
    MatrixDense<double> mat_dense(make_norm_A(m, m));
    std::cout << mat_dense.get_info_string() << std::endl;

    NoFillMatrixSparse<__half> mat_dense_hlf = mat_dense.cast<__half>();
    std::cout << mat_dense_hlf.get_info_string() << std::endl;

    // std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [this] (
    //     NoFillMatrixSparse<double> &A, Vector<double> &x_temp
    // ) {

    //     Vector<double> b = A*x_temp;

    //     GenericLinearSystem<NoFillMatrixSparse> lin_sys(A, b);
    //     SolveArgPkg args(nested_outer_iter, nested_inner_iter, 0.);

    //     SimpleConstantThreshold<NoFillMatrixSparse> mp_restarted_gmres(lin_sys, args);

    //     mp_restarted_gmres.solve();
    //     std::cout << mp_restarted_gmres.get_info_string() << std::endl;

    // };

    // basic_func_benchmark<NoFillMatrixSparse, double>(
    //     nested_n_min, nested_n_max, make_norm_A, execute_func, "mp_gmres_ir"
    // );

}
// #include <functional>
// #include <fstream>

// #include <cmath>

// #include "types/types.h"

// #include "../test.h"

// #include "include/benchmark_Sparse.h"

// class NoFillMatrixSparse_Benchmark: public Benchmark_Sparse
// {
// public:

//     int min_n_mult = 8;
//     int max_n_mult = 15-prototying_n_speed_up;

//     int min_n_substitution = 7;
//     int max_n_substitution = 13-prototying_n_speed_up;

//     std::function<NoFillMatrixSparse<double> (int, int)> make_low_tri_A = [this] (
//         int m, int n
//     ) -> NoFillMatrixSparse<double> {
//         return make_lower_tri(make_A(m, n));
//     };
    
//     std::function<NoFillMatrixSparse<double> (int, int)> make_upp_tri_A = [this] (
//         int m, int n
//     ) -> NoFillMatrixSparse<double> {
//         return make_upper_tri(make_A(m, n));
//     };

// };

// TEST_F(NoFillMatrixSparse_Benchmark, MatrixVectorMult_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A*b;
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_mult, max_n_mult, make_A, execute_func, "matsparse_mv"
//     );

// }

// TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixVectorMult_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.transpose_prod(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_mult, max_n_mult, make_A, execute_func, "matsparse_tmv"
//     );

// }

// TEST_F(NoFillMatrixSparse_Benchmark, MatrixBlockVectorMult_BENCHMARK) {
// }

// TEST_F(NoFillMatrixSparse_Benchmark, TransposeMatrixBlockVectorMult_BENCHMARK) {
// }

// TEST_F(NoFillMatrixSparse_Benchmark, ForwardSubstitution_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.frwd_sub(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_substitution, max_n_substitution, make_low_tri_A, execute_func, "matsparse_frwdsub"
//     );

// }

// TEST_F(NoFillMatrixSparse_Benchmark, BackwardSubstitution_BENCHMARK) {

//     std::function<void (NoFillMatrixSparse<double> &, Vector<double> &)> execute_func = [] (
//         NoFillMatrixSparse<double> &A, Vector<double> &b
//     ) {
//         A.back_sub(b);
//     };

//     basic_func_benchmark<NoFillMatrixSparse>(
//         min_n_substitution, max_n_substitution, make_upp_tri_A, execute_func, "matsparse_backsub"
//     );

// }
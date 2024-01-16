// #include <filesystem>
// #include <memory>
// #include <iostream>
// #include <cmath>
// #include <string>

// #include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
// #include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

// #include "types/types.h"
// #include "tools/tools.h"

// namespace fs = std::filesystem;
// using std::shared_ptr, std::make_shared;
// using std::cout, std::endl;
// using std::pow;
// using std::string;

// const double u_hlf = pow(2, -10);
// const double u_sgl = pow(2, -23);
// const double u_dbl = pow(2, -52);

// string get_file_name(fs::path file_path) {

//     string temp = file_path.string();
//     temp = temp.substr(temp.find_last_of("//")+1);
//     temp = temp.substr(0, temp.find_last_of("."));

//     return temp;

// }

// template <template <typename> typename M>
// void print_solver_info(
//     shared_ptr<GenericIterativeSolve<M>> solver,
//     string ID
// ) {
//     cout << "Name: " << ID << " | ";
//     cout << "Converged: " << solver->check_converged() << " | ";
//     cout << "Iter: " << solver->get_iteration() << " | ";
//     cout << "Relres: " << solver->get_relres() << endl;
// }

// int main() {

//     cout << "*** STARTING NUMERICAL EXPERIMENTATION ***" << endl;

//     fs::path load_dir("/home/bdosre/dev/numerical_experimentation/data/experiment_matrices");
//     fs::path save_dir("/home/bdosre/dev/numerical_experimentation/data/0_2_50");

//     fs::directory_iterator iter(load_dir);
//     fs::directory_iterator curr = fs::begin(iter);

//     for (fs::directory_iterator curr = fs::begin(iter); curr != fs::end(iter); ++curr) {

//         MatrixDense<double> A_dense = read_matrixCSV<MatrixDense, double>(*curr);
//         A_dense = 1/(A_dense.maxCoeff())*A_dense;
//         MatrixSparse<double> A = A_dense.sparseView();

//         cout << "Testing: " << *curr << " of size " << A.rows() << "x" << A.cols() << endl;

//         SolveArgPkg args;
//         args.init_guess = MatrixVector<double>::Zero(A.cols());
//         args.max_iter = 50;
//         args.max_inner_iter = static_cast<int>(0.2*A.rows());
//         args.target_rel_res = pow(10, -10);

//         for (int i=1; i<=3; ++i) {

//             string ID_prefix = get_file_name(*curr) + "_" + to_string(i);
//             MatrixVector<double> b = A*MatrixVector<double>::Random(A.cols());

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_hlf = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, half>>(A, b, u_hlf, args)
//             );
//             fpgmres_hlf->solve();
//             record_solve(fpgmres_hlf,
//                          save_dir / fs::path(ID_prefix+"_fphlf.json"),
//                          ID_prefix+"_fphlf");
//             print_solver_info(fpgmres_hlf, ID_prefix+"_fphlf");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_sgl = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, float>>(A, b, u_sgl, args)
//             );
//             fpgmres_sgl->solve();
//             record_solve(fpgmres_sgl,
//                          save_dir / fs::path(ID_prefix+"_fpsgl.json"),
//                          ID_prefix+"_fpsgl");
//             print_solver_info(fpgmres_sgl, ID_prefix+"_fpsgl");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_dbl = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, double>>(A, b, u_dbl, args)
//             );
//             fpgmres_dbl->solve();
//             record_solve(fpgmres_dbl,
//                          save_dir / fs::path(ID_prefix+"_fpdbl.json"),
//                          ID_prefix+"_fpdbl");
//             print_solver_info(fpgmres_dbl, ID_prefix+"_fpdbl");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> mpgmres = (
//                 make_shared<SimpleConstantThreshold<MatrixSparse>>(A, b, args)
//             );
//             mpgmres->solve();
//             record_solve(mpgmres,
//                          save_dir / fs::path(ID_prefix+"_mp.json"),
//                          ID_prefix+"_mp");
//             print_solver_info(mpgmres, ID_prefix+"_mp");

//         }

//     }

//     cout << "*** FINISH NUMERICAL EXPERIMENTATION ***" << endl;
    
//     return 0;

// }
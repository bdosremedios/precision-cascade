// #ifndef EXPERIMENT_RECORD_H
// #define EXPERIMENT_RECORD_H

// #include "experiment_log.h"
// #include "experiment_tools.h"

// #include "solvers/IterativeSolve.h"
// #include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

// #include <iostream>
// #include <iomanip>
// #include <fstream>
// #include <filesystem>
// #include <string>
// #include <chrono>

// namespace fs = std::filesystem;

// using namespace cascade;

// std::string bool_to_string(bool b);

// template <template <typename> typename TMatrix>
// void record_basic_solver_data(
//     std::ofstream &file_out,
//     const std::string ID,
//     const std::shared_ptr<GenericIterativeSolve<TMatrix>> &solver_ptr,
//     const Experiment_Clock &clock
// ) {
//     file_out << "\t\"id\" : \"" << ID << "\",\n";
//     file_out << "\t\"solver_class\" : \"" << typeid(*solver_ptr).name()
//              << "\",\n";
//     file_out << "\t\"initiated\" : \""
//              << bool_to_string(solver_ptr->check_initiated())
//              << "\",\n";
//     file_out << "\t\"converged\" : \""
//              << bool_to_string(solver_ptr->check_converged())
//              << "\",\n";
//     file_out << "\t\"terminated\" : \""
//              << bool_to_string(solver_ptr->check_terminated())
//              << "\",\n";
//     file_out << "\t\"iteration\" : " << solver_ptr->get_iteration()
//              << ",\n";
//     file_out << "\t\"elapsed_time_ms\" : " << clock.get_elapsed_time_ms()
//              << ",\n";
// }


// std::string vector_to_jsonarray_str(std::vector<double> vec, int padding_level);

// template <template <typename> typename TMatrix>
// void record_residual_solver_data(
//     std::ofstream &file_out,
//     const std::shared_ptr<GenericIterativeSolve<TMatrix>> &solver_ptr,
//     const int padding
// ) {
//     file_out << "\t\"res_norm_history\" : "
//              << vector_to_jsonarray_str(
//                     solver_ptr->get_res_norm_history(),
//                     padding
//                 )
//              << "\n";
// }

// std::ofstream open_json_ofstream(
//     std::string file_name, fs::path save_dir, Experiment_Log logger
// );

// void start_json(std::ofstream &file_out);

// void end_json(std::ofstream &file_out);

// template <template <typename> typename TMatrix, typename TPrecision>
// void record_FPGMRES_data_json(
//     const Solve_Data<GenericIterativeSolve, TMatrix> &data,
//     const PrecondArgPkg<TMatrix, TPrecision> arg_precond_arg_pkg,
//     const std::string file_name,
//     const fs::path save_dir,
//     Experiment_Log logger
// ) {
    
//     std::ofstream file_out = open_json_ofstream(file_name, save_dir, logger);

//     start_json(file_out);

//     record_basic_solver_data<TMatrix>(
//         file_out, file_name, data.solver_ptr, data.clock
//     );
//     record_residual_solver_data<TMatrix>(
//         file_out, data.solver_ptr, 0
//     );

//     end_json(file_out);

// }

// template <template <typename> typename TMatrix>
// void record_MPGMRES_data_json(
//     const Solve_Data<MP_GMRES_IR_Solve, TMatrix> &data,
//     const PrecondArgPkg<TMatrix, double> arg_precond_arg_pkg,
//     const std::string file_name,
//     const fs::path save_dir,
//     Experiment_Log logger
// ) {

//     std::ofstream file_out = open_json_ofstream(file_name, save_dir, logger);

//     start_json(file_out);

//     record_basic_solver_data<TMatrix>(
//         file_out, file_name, data.solver_ptr, data.clock
//     );
//     file_out << "\t\"hlf_sgl_cascade_change\" : "
//                 << data.solver_ptr->get_hlf_sgl_cascade_change()
//                 << ",\n";
//     file_out << "\t\"sgl_dbl_cascade_change\" : "
//                 << data.solver_ptr->get_sgl_dbl_cascade_change()
//                 << ",\n";
//     record_residual_solver_data<TMatrix>(
//         file_out, data.solver_ptr, 0
//     );

//     end_json(file_out);

// }

// #endif
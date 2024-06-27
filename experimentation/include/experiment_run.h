#ifndef EXPERIMENT_RUN_H
#define EXPERIMENT_RUN_H

#include "experiment_log.h"
#include "experiment_read.h"
#include "experiment_record.h"

#include "tools/cuHandleBundle.h"
#include "types/types.h"
#include "tools/read_matrix.h"
#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

namespace fs = std::filesystem;

using namespace cascade;

static const double u_hlf = std::pow(2, -10);
static const double u_sgl = std::pow(2, -23);
static const double u_dbl = std::pow(2, -52);

template <template <typename> typename TMatrix>
GenericLinearSystem<TMatrix> load_lin_sys(
    const cuHandleBundle &cu_handles,
    fs::path input_dir,
    std::string matrix_name,
    Experiment_Log logger
) {

    fs::path matrix_path = input_dir / fs::path(matrix_name);

    logger.info("Loading: "+matrix_path.string());

    TMatrix<double> A(cu_handles);
    if (matrix_path.extension() == ".mtx") {
        A = read_matrixMTX<TMatrix, double>(cu_handles, matrix_path);
    } else if (matrix_path.extension() == ".csv") {
        A = read_matrixCSV<TMatrix, double>(cu_handles, matrix_path);
    } else {
        throw std::runtime_error("load_lin_sys: invalid extension");
    }

    Scalar<double> A_max_mag = A.get_max_mag_elem();
    A /= A_max_mag;
    logger.info("Matrix info: " + A.get_info_string());

    // Search for a rhs and if none is found generate one randomly
    fs::path potential_b_path(
        input_dir /
        fs::path(
            matrix_path.stem().string() + "_b" +
            matrix_path.extension().string()
        )
    );
    Vector<double> b(cu_handles);
    if (fs::exists(potential_b_path)) {
        if (potential_b_path.extension() == ".mtx") {
            b = read_vectorMTX<double>(cu_handles, potential_b_path, "random");
        } else if (potential_b_path.extension() == ".csv") {
            b = read_vectorCSV<double>(cu_handles, potential_b_path);
        } else {
            throw std::runtime_error(
                "load_lin_sys: invalid extension found on potential_b_path file"
            );
        }
        b /= A_max_mag;
    } else {
        b = A*Vector<double>::Random(cu_handles, A.cols());
    }

    return GenericLinearSystem<TMatrix>(A, b);

}

void create_or_clear_directory(fs::path dir, Experiment_Log logger);

template <
    template <template <typename> typename> typename TSolver,
    template <typename> typename TMatrix
>
Experiment_Data<TSolver, TMatrix> execute_solve(
    std::shared_ptr<TSolver<TMatrix>> arg_solver_ptr,
    bool show_plots
) {

    Experiment_Clock exp_clock;
    exp_clock.start_clock_experiment();
    arg_solver_ptr->solve();
    if (show_plots) { arg_solver_ptr->view_relres_plot("log"); }
    exp_clock.stop_clock_experiment();

    return Experiment_Data<TSolver, TMatrix>(exp_clock, arg_solver_ptr);

}

template <template <typename> typename TMatrix, typename TPrecision>
void run_record_FPGMRES_solve(
    std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, TPrecision>> arg_solver_ptr,
    PrecondArgPkg<TMatrix, TPrecision> arg_precond_arg_pkg,
    std::string matrix_name,
    std::string solve_name,
    std::string precond_name,
    int exp_iter,
    fs::path output_dir,
    bool show_plots,
    Experiment_Log logger
) {
    std::string solve_experiment_id = (
        matrix_name + "_" +
        solve_name + "_" +
        precond_name + "_" +
        std::to_string(exp_iter)
    );
    logger.info("Running solve experiment: " + solve_experiment_id);
    Experiment_Data<GenericIterativeSolve, TMatrix> data = (
        execute_solve<GenericIterativeSolve, TMatrix>(
            arg_solver_ptr,
            show_plots
        )
    );
    logger.info(data.get_info_string());
    record_FPGMRES_data_json(
        data,
        arg_precond_arg_pkg,
        precond_name,
        solve_experiment_id,
        output_dir,
        logger
    );
}

template <template <typename> typename TMatrix>
void run_record_MPGMRES_solve(
    std::shared_ptr<MP_GMRES_IR_Solve<TMatrix>> arg_solver_ptr,
    PrecondArgPkg<TMatrix, double> arg_precond_arg_pkg,
    std::string matrix_name,
    std::string solve_name,
    std::string precond_name,
    int exp_iter,
    fs::path output_dir,
    bool show_plots,
    Experiment_Log logger
) {
    std::string solve_experiment_id = (
        matrix_name + "_" +
        solve_name + "_" +
        precond_name + "_" +
        std::to_string(exp_iter)
    );
    logger.info("Running solve experiment: " + solve_experiment_id);
    Experiment_Data<MP_GMRES_IR_Solve, TMatrix> data = (
        execute_solve<MP_GMRES_IR_Solve, TMatrix>(
            arg_solver_ptr,
            show_plots
        )
    );
    logger.info(data.get_info_string());
    record_MPGMRES_data_json(
        data,
        arg_precond_arg_pkg,
        precond_name,
        solve_experiment_id,
        output_dir,
        logger
    );
}

template <template <typename> typename TMatrix>
void run_solve_group(
    const cuHandleBundle &cu_handles,
    Solve_Group solve_group,
    fs::path data_dir,
    fs::path output_dir,
    Experiment_Log outer_logger
) {

    outer_logger.info("Running solve group: "+solve_group.id);

    fs::path solve_group_dir = output_dir / fs::path(solve_group.id);
    create_or_clear_directory(solve_group_dir, outer_logger);

    Experiment_Log logger(
        solve_group.id + "_logger", solve_group_dir /
        fs::path(solve_group.id + ".log"), false
    );
    logger.info("Solve info: " + solve_group.solver_args.get_info_string());

    bool show_plots = false;

    for (std::string matrix_name : solve_group.matrices_to_test) {
        for (
            int exp_iter = 0;
            exp_iter < solve_group.experiment_iterations;
            ++exp_iter
        ) {

            // Load linear system, generating b to solve
            GenericLinearSystem<TMatrix> gen_lin_sys = load_lin_sys<TMatrix>(
                cu_handles, data_dir, matrix_name, logger
            );

            // Determine preconditioning
            PrecondArgPkg<TMatrix, double> precond_args_dbl;
            if (solve_group.precond_specs.name == "none") {
                logger.info("Preconditioner: NoPreconditioner");
                precond_args_dbl = PrecondArgPkg<TMatrix, double>(
                    std::make_shared<NoPreconditioner<TMatrix, double>>()
                );
            } else if (solve_group.precond_specs.name == "jacobi") {
                logger.info("Preconditioner: JacobiPreconditioner");
                precond_args_dbl = PrecondArgPkg<TMatrix, double>(
                    std::make_shared<JacobiPreconditioner<TMatrix, double>>(
                        gen_lin_sys.get_A()
                    )
                );
            } else if (solve_group.precond_specs.name == "ilu0") {
                logger.info("Preconditioner: ILU(0) starting computation");
                std::shared_ptr<ILUPreconditioner<TMatrix, double>> ilu0 = (
                    std::make_shared<ILUPreconditioner<TMatrix, double>>(
                        gen_lin_sys.get_A()
                    )
                );
                logger.info("Preconditioner: ILU(0) finished computation");
                logger.info(
                    "Preconditioner: L info: " +
                    ilu0->get_L().get_info_string()
                );
                logger.info(
                    "Preconditioner: U info: " +
                    ilu0->get_U().get_info_string()
                );
                precond_args_dbl = PrecondArgPkg<TMatrix, double>(ilu0);
            } else if (solve_group.precond_specs.name == "ilutp") {
                std::stringstream ilutp_strm;
                ilutp_strm << std::setprecision(3);
                ilutp_strm << "ILUTP("
                           << solve_group.precond_specs.ilutp_tau << ", "
                           << solve_group.precond_specs.ilutp_p << ")";
                logger.info(
                    "Preconditioner: {} starting computation" + ilutp_strm.str()
                );
                std::shared_ptr<ILUPreconditioner<TMatrix, double>> ilutp = (
                    std::make_shared<ILUPreconditioner<TMatrix, double>>(
                        gen_lin_sys.get_A(),
                        solve_group.precond_specs.ilutp_tau,
                        solve_group.precond_specs.ilutp_p,
                        true
                    )
                );
                logger.info(
                    "Preconditioner: {} finished computation" + ilutp_strm.str()
                );
                logger.info(
                    "Preconditioner: L info: "+ilutp->get_L().get_info_string()
                );
                logger.info(
                    "Preconditioner: U info: "+ilutp->get_U().get_info_string()
                );
                logger.info(
                    "Preconditioner: P info: "+ilutp->get_P().get_info_string()
                );
                precond_args_dbl = PrecondArgPkg<TMatrix, double>(ilutp);
            } else {
                throw std::runtime_error(
                    "run_solve_group: invalid precond_specs encountered in "
                    "\"" + solve_group.id + "\""
                ); 
            }
            
            // Run solves
            for (std::string solver_id : solve_group.solvers_to_use) {

                if (solver_id == "FP16") {

                    PrecondArgPkg<TMatrix, __half> * precond_args_hlf_ptr = (
                        precond_args_dbl.cast_hlf_ptr()
                    );
                    TypedLinearSystem<TMatrix, __half> lin_sys_hlf(
                        &gen_lin_sys
                    );
                    run_record_FPGMRES_solve<TMatrix, __half>(
                        std::make_shared<FP_GMRES_IR_Solve<TMatrix, __half>>(
                            &lin_sys_hlf, u_hlf,
                            solve_group.solver_args, *precond_args_hlf_ptr
                        ),
                        *precond_args_hlf_ptr,
                        matrix_name,
                        solver_id,
                        solve_group.precond_specs.get_spec_string(),
                        exp_iter,
                        solve_group_dir,
                        false, logger
                    );
                    delete precond_args_hlf_ptr;

                } else if (solver_id == "FP32") {

                    PrecondArgPkg<TMatrix, float> * precond_args_sgl_ptr = (
                        precond_args_dbl.cast_sgl_ptr()
                    );
                    TypedLinearSystem<TMatrix, float> lin_sys_sgl(
                        &gen_lin_sys
                    );
                    run_record_FPGMRES_solve<TMatrix, float>(
                        std::make_shared<FP_GMRES_IR_Solve<TMatrix, float>>(
                            &lin_sys_sgl, u_sgl,
                            solve_group.solver_args, *precond_args_sgl_ptr
                        ),
                        *precond_args_sgl_ptr,
                        matrix_name,
                        solver_id,
                        solve_group.precond_specs.get_spec_string(),
                        exp_iter,
                        solve_group_dir,
                        false, logger
                    );
                    delete precond_args_sgl_ptr;

                } else if (solver_id == "FP64") {

                    TypedLinearSystem<TMatrix, double> lin_sys_dbl(
                        &gen_lin_sys
                    );
                    run_record_FPGMRES_solve<TMatrix, double>(
                        std::make_shared<FP_GMRES_IR_Solve<TMatrix, double>>(
                            &lin_sys_dbl, u_dbl,
                            solve_group.solver_args, precond_args_dbl
                        ),
                        precond_args_dbl,
                        matrix_name,
                        solver_id,
                        solve_group.precond_specs.get_spec_string(),
                        exp_iter,
                        solve_group_dir,
                        false, logger
                    );

                } else if (solver_id == "SimpleConstantThreshold") {

                    run_record_MPGMRES_solve<TMatrix>(
                        std::make_shared<SimpleConstantThreshold<TMatrix>>(
                            &gen_lin_sys,
                            solve_group.solver_args, precond_args_dbl
                        ),
                        precond_args_dbl,
                        matrix_name,
                        solver_id,
                        solve_group.precond_specs.get_spec_string(),
                        exp_iter,
                        solve_group_dir,
                        false, logger
                    );

                } else if (solver_id == "RestartCount") {

                    run_record_MPGMRES_solve<TMatrix>(
                        std::make_shared<RestartCount<TMatrix>>(
                            &gen_lin_sys,
                            solve_group.solver_args, precond_args_dbl
                        ),
                        precond_args_dbl,
                        matrix_name,
                        solver_id,
                        solve_group.precond_specs.get_spec_string(),
                        exp_iter,
                        solve_group_dir,
                        false, logger
                    );

                } else {
                    throw std::runtime_error(
                        "run_solve_group: invalid solver id encountered "
                        "\"" + solve_group.id + "\""
                    ); 
                }

            }

        }
    }

}

void run_experimental_spec(
    const cuHandleBundle &cu_handles,
    Experiment_Specification exp_spec,
    fs::path data_dir,
    fs::path output_dir,
    Experiment_Log logger
);

#endif
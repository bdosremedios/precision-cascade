#ifndef EXPERIMENT_RUN_H
#define EXPERIMENT_RUN_H

#include "exp_read/exp_read.h"
#include "exp_record/exp_record.h"
#include "exp_tools/exp_tools.h"

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

void create_or_clear_directory(fs::path dir, Experiment_Log logger);

template <template <typename> typename TMatrix>
Preconditioner_Data<TMatrix> calc_preconditioner(
    const GenericLinearSystem<TMatrix> &gen_lin_sys,
    Preconditioner_Spec precond_specs,
    Experiment_Log logger
) {

    Experiment_Clock exp_clock;

    PrecondArgPkg<TMatrix, double> precond_args_dbl;

    logger.info("Start Precond Calc: " + precond_specs.get_spec_string());

    if (precond_specs.name == "none") {

        exp_clock.start_clock_experiment();
        precond_args_dbl = PrecondArgPkg<TMatrix, double>(
            std::make_shared<NoPreconditioner<TMatrix, double>>()
        );
        exp_clock.stop_clock_experiment();

    } else if (precond_specs.name == "jacobi") {

        exp_clock.start_clock_experiment();
        precond_args_dbl = PrecondArgPkg<TMatrix, double>(
            std::make_shared<JacobiPreconditioner<TMatrix, double>>(
                gen_lin_sys.get_A()
            )
        );
        exp_clock.stop_clock_experiment();

    } else if (precond_specs.name == "ilu0") {

        exp_clock.start_clock_experiment();
        std::shared_ptr<ILUPreconditioner<TMatrix, double>> ilu0 = (
            std::make_shared<ILUPreconditioner<TMatrix, double>>(
                gen_lin_sys.get_A()
            )
        );
        exp_clock.stop_clock_experiment();

        logger.info("Precond: L info: " + ilu0->get_L().get_info_string());
        logger.info("Precond: U info: " + ilu0->get_U().get_info_string());
        precond_args_dbl = PrecondArgPkg<TMatrix, double>(ilu0);

    } else if (precond_specs.name == "ilutp") {

        exp_clock.start_clock_experiment();
        std::shared_ptr<ILUPreconditioner<TMatrix, double>> ilutp = (
            std::make_shared<ILUPreconditioner<TMatrix, double>>(
                gen_lin_sys.get_A(),
                precond_specs.ilutp_tau,
                precond_specs.ilutp_p,
                true
            )
        );
        exp_clock.stop_clock_experiment();

        logger.info("Precond: L info: " + ilutp->get_L().get_info_string());
        logger.info("Precond: U info: " + ilutp->get_U().get_info_string());
        logger.info("Precond: P info: " + ilutp->get_P().get_info_string());
        precond_args_dbl = PrecondArgPkg<TMatrix, double>(ilutp);

    } else {
        throw std::runtime_error(
            "calc_preconditioner: invalid precond_specs encountered"
        );
    }

    logger.info("Finished Precond Calc");

    return Preconditioner_Data<TMatrix>(
        precond_specs.get_spec_string(),
        exp_clock,
        precond_specs,
        precond_args_dbl
    );

}

template <
    template <template <typename> typename> typename TSolver,
    template <typename> typename TMatrix
>
Solve_Data<TSolver, TMatrix> execute_solve(
    std::string data_id,
    std::shared_ptr<TSolver<TMatrix>> arg_solver_ptr,
    bool show_plots
) {

    Experiment_Clock exp_clock;
    exp_clock.start_clock_experiment();
    arg_solver_ptr->solve();
    if (show_plots) { arg_solver_ptr->view_relres_plot("log"); }
    exp_clock.stop_clock_experiment();

    return Solve_Data<TSolver, TMatrix>(data_id, exp_clock, arg_solver_ptr);

}

template <template <typename> typename TMatrix, typename TPrecision>
void run_record_FPGMRES_solve(
    std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, TPrecision>> arg_solver_ptr,
    PrecondArgPkg<TMatrix, TPrecision> arg_precond_arg_pkg,
    std::string solver_name,
    fs::path output_data_dir,
    Experiment_Log logger
) {
    logger.info("Running solve experiment: " + solver_name);
    Solve_Data<GenericIterativeSolve, TMatrix> data = (
        execute_solve<GenericIterativeSolve, TMatrix>(
            solver_name, arg_solver_ptr, false
        )
    );
    logger.info(data.get_info_string());
    data.record_json(solver_name, output_data_dir, logger);
}

template <template <typename> typename TMatrix>
void run_record_MPGMRES_solve(
    std::shared_ptr<MP_GMRES_IR_Solve<TMatrix>> arg_solver_ptr,
    PrecondArgPkg<TMatrix, double> arg_precond_arg_pkg,
    std::string solver_name,
    fs::path output_data_dir,
    Experiment_Log logger
) {
    logger.info("Running solve experiment: " + solver_name);
    Solve_Data<MP_GMRES_IR_Solve, TMatrix> data = (
        execute_solve<MP_GMRES_IR_Solve, TMatrix>(
            solver_name, arg_solver_ptr, false
        )
    );
    logger.info(data.get_info_string());
    data.record_json(solver_name, output_data_dir, logger);
}

template <template <typename> typename TMatrix>
void run_record_solversuite_experiment(
    const GenericLinearSystem<TMatrix> &gen_lin_sys,
    Solve_Group solve_group,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    // Determine preconditioning
    Preconditioner_Data<TMatrix> precond_data = calc_preconditioner<TMatrix>(
        gen_lin_sys, solve_group.precond_specs, logger
    );
    precond_data.record_json("preconditioner", output_data_dir, logger);

    // Run solves
    for (std::string solver_id : solve_group.solvers_to_use) {

        if (solver_id == "FP16") {

            cascade::PrecondArgPkg<TMatrix, __half> * precond_args_hlf_ptr = (
                precond_data.precond_arg_pkg_dbl.cast_hlf_ptr()
            );
            cascade::TypedLinearSystem<TMatrix, __half> lin_sys_hlf(
                &gen_lin_sys
            );
            run_record_FPGMRES_solve<TMatrix, __half>(
                std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, __half>>(
                    &lin_sys_hlf,
                    u_hlf,
                    solve_group.solver_args,
                    *precond_args_hlf_ptr
                ),
                *precond_args_hlf_ptr,
                solver_id,
                output_data_dir,
                logger
            );
            delete precond_args_hlf_ptr;

        } else if (solver_id == "FP32") {

            cascade::PrecondArgPkg<TMatrix, float> * precond_args_sgl_ptr = (
                precond_data.precond_arg_pkg_dbl.cast_sgl_ptr()
            );
            cascade::TypedLinearSystem<TMatrix, float> lin_sys_sgl(
                &gen_lin_sys
            );
            run_record_FPGMRES_solve<TMatrix, float>(
                std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, float>>(
                    &lin_sys_sgl,
                    u_sgl,
                    solve_group.solver_args,
                    *precond_args_sgl_ptr
                ),
                *precond_args_sgl_ptr,
                solver_id,
                output_data_dir,
                logger
            );
            delete precond_args_sgl_ptr;

        } else if (solver_id == "FP64") {

            cascade::TypedLinearSystem<TMatrix, double> lin_sys_dbl(
                &gen_lin_sys
            );
            run_record_FPGMRES_solve<TMatrix, double>(
                std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, double>>(
                    &lin_sys_dbl,
                    u_dbl,
                    solve_group.solver_args,
                    precond_data.precond_arg_pkg_dbl
                ),
                precond_data.precond_arg_pkg_dbl,
                solver_id,
                output_data_dir,
                logger
            );

        } else if (solver_id == "SimpleConstantThreshold") {

            run_record_MPGMRES_solve<TMatrix>(
                std::make_shared<cascade::SimpleConstantThreshold<TMatrix>>(
                    &gen_lin_sys,
                    solve_group.solver_args,
                    precond_data.precond_arg_pkg_dbl
                ),
                precond_data.precond_arg_pkg_dbl,
                solver_id,
                output_data_dir,
                logger
            );

        } else if (solver_id == "RestartCount") {

            run_record_MPGMRES_solve<TMatrix>(
                std::make_shared<cascade::RestartCount<TMatrix>>(
                    &gen_lin_sys,
                    solve_group.solver_args,
                    precond_data.precond_arg_pkg_dbl
                ),
                precond_data.precond_arg_pkg_dbl,
                solver_id,
                output_data_dir,
                logger
            );

        } else {
            throw std::runtime_error(
                "run_record_solve_group_iteration: invalid solver_id "
                "encountered \"" + solve_group.id + "\""
            ); 
        }

    }

}

template <template <typename> typename TMatrix>
void run_record_solve_group(
    const cuHandleBundle &cu_handles,
    Solve_Group solve_group,
    fs::path matrix_data_dir,
    fs::path output_data_dir,
    Experiment_Log outer_logger
) {

    outer_logger.info("Running solve group: "+solve_group.id);

    fs::path solve_group_dir = output_data_dir / fs::path(solve_group.id);
    create_or_clear_directory(solve_group_dir, outer_logger);

    Experiment_Log solve_group_logger(
        solve_group.id + "_logger", solve_group_dir /
        fs::path(solve_group.id + ".log"), false
    );
    solve_group_logger.info(
        "Solve info: " + solve_group.solver_args.get_info_string()
    );

    // Iterate over matrices and iterations per matrix
    for (std::string matrix_file : solve_group.matrices_to_test) {

        fs::path matrix_output_data_dir = (
            solve_group_dir / fs::path(matrix_file).stem()
        );
        create_or_clear_directory(matrix_output_data_dir, solve_group_logger);
        
        int total_iters = solve_group.experiment_iterations;
        for (int exp_iter = 0; exp_iter < total_iters; ++exp_iter) {

            fs::path iter_output_data_dir = (
                matrix_output_data_dir / fs::path(std::to_string(exp_iter))
            );
            create_or_clear_directory(iter_output_data_dir, solve_group_logger);

            // Load linear system, generating b to solve
            GenericLinearSystem<TMatrix> gen_lin_sys = load_lin_sys<TMatrix>(
                cu_handles, matrix_data_dir, matrix_file, solve_group_logger
            );

            run_record_solversuite_experiment(
                gen_lin_sys,
                solve_group,
                iter_output_data_dir,
                solve_group_logger
            );

        }

    }
}

void run_record_experimental_spec(
    const cuHandleBundle &cu_handles,
    Experiment_Spec exp_spec,
    fs::path matrix_data_dir,
    fs::path output_data_dir,
    Experiment_Log logger
);

#endif
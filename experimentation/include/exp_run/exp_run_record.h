#ifndef EXP_RUN_RECORD_H
#define EXP_RUN_RECORD_H

#include "exp_read/exp_read.h"
#include "exp_data/exp_data.h"
#include "exp_tools/exp_tools.h"
#include "exp_generate_data.h"

#include "tools/cuHandleBundle.h"
#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include <cmath>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static const double u_hlf = std::pow(2, -10);
static const double u_sgl = std::pow(2, -23);
static const double u_dbl = std::pow(2, -52);

void create_or_clear_directory(fs::path dir, Experiment_Log logger);

template <template <typename> typename TMatrix>
void run_record_fpgmres_solve(
    std::string solver_id,
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
    cascade::SolveArgPkg solve_arg_pkg,
    cascade::PrecondArgPkg<TMatrix, double> precond_arg_pkg_dbl,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    Solve_Data<cascade::GenericIterativeSolve, TMatrix> data;

    if (solver_id == "FP16") {

        cascade::PrecondArgPkg<TMatrix, __half> * precond_args_hlf_ptr = (
            precond_arg_pkg_dbl.cast_hlf_ptr()
        );
        cascade::TypedLinearSystem<TMatrix, __half> lin_sys_hlf(&gen_lin_sys);
        data = execute_solve<cascade::GenericIterativeSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, __half>>(
                &lin_sys_hlf, u_hlf, solve_arg_pkg, *precond_args_hlf_ptr
            ),
            false
        );
        delete precond_args_hlf_ptr;

    } else if (solver_id == "FP32") {

        cascade::PrecondArgPkg<TMatrix, float> * precond_args_sgl_ptr = (
            precond_arg_pkg_dbl.cast_sgl_ptr()
        );
        cascade::TypedLinearSystem<TMatrix, float> lin_sys_sgl(&gen_lin_sys);
        data = execute_solve<cascade::GenericIterativeSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, float>>(
                &lin_sys_sgl, u_sgl, solve_arg_pkg, *precond_args_sgl_ptr
            ),
            false
        );
        delete precond_args_sgl_ptr;

    } else if (solver_id == "FP64") {

        cascade::TypedLinearSystem<TMatrix, double> lin_sys_dbl(&gen_lin_sys);
        data = execute_solve<cascade::GenericIterativeSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, double>>(
                &lin_sys_dbl, u_dbl, solve_arg_pkg, precond_arg_pkg_dbl
            ),
            false
        );

    } else {

        std::runtime_error(
            "run_record_fpgmres_solve: invalid fixed precision (FP) solver_id "
            "encountered"
        );

    }

    logger.info(data.get_info_string());
    data.record_json(solver_id, output_data_dir, logger);

}

template <template <typename> typename TMatrix>
void run_record_mpgmres_solve(
    std::string solver_id,
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
    cascade::SolveArgPkg solve_arg_pkg,
    cascade::PrecondArgPkg<TMatrix, double> precond_arg_pkg_dbl,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    Solve_Data<cascade::MP_GMRES_IR_Solve, TMatrix> data;
    std::shared_ptr<cascade::MP_GMRES_IR_Solve<TMatrix>> solver_ptr;
    
    if (solver_id == "SimpleConstantThreshold") {

        solver_ptr = (
            std::make_shared<cascade::SimpleConstantThreshold<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );

    } else if (solver_id == "RestartCount") {

        solver_ptr = (
            std::make_shared<cascade::RestartCount<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );

    } else {

        std::runtime_error(
            "run_record_mpgmres_solve: invalid mixed precision solver_id "
            "encountered"
        );

    }

    data = execute_solve<cascade::MP_GMRES_IR_Solve, TMatrix>(
        solver_id, solver_ptr, false
    );

    logger.info(data.get_info_string());
    data.record_json(solver_id, output_data_dir, logger);

}

template <template <typename> typename TMatrix>
void run_record_solversuite_experiment(
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
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

        logger.info("Running solve experiment: " + solver_id);

        if (Solve_Group::valid_fp_solver_ids.count(solver_id) == 1) {

            run_record_fpgmres_solve(
                solver_id,
                gen_lin_sys,
                solve_group.solver_args,
                precond_data.precond_arg_pkg_dbl,
                output_data_dir,
                logger
            );

        } else if (Solve_Group::valid_mp_solver_ids.count(solver_id) == 1) {

            run_record_mpgmres_solve(
                solver_id,
                gen_lin_sys,
                solve_group.solver_args,
                precond_data.precond_arg_pkg_dbl,
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
    const cascade::cuHandleBundle &cu_handles,
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
            cascade::GenericLinearSystem<TMatrix> gen_lin_sys = (
                load_lin_sys<TMatrix>(
                    cu_handles, matrix_data_dir, matrix_file, solve_group_logger
                )
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
    const cascade::cuHandleBundle &cu_handles,
    Experiment_Spec exp_spec,
    fs::path matrix_data_dir,
    fs::path output_data_dir,
    Experiment_Log logger
);

#endif
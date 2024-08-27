#ifndef EXP_RUN_RECORD_H
#define EXP_RUN_RECORD_H

#include "exp_read/exp_read.h"
#include "exp_data/exp_data.h"
#include "exp_tools/exp_tools.h"
#include "exp_generate_data.h"

#include "tools/cuHandleBundle.h"
#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/VP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include <cmath>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static const double u_hlf = std::pow(2, -10);
static const double u_sgl = std::pow(2, -23);
static const double u_dbl = std::pow(2, -52);

void create_directory_if_nexists(fs::path dir, Experiment_Log logger);

template <template <typename> typename TMatrix>
void run_record_fpgmres_solve(
    std::string solver_id,
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
    cascade::SolveArgPkg solve_arg_pkg,
    cascade::PrecondArgPkg<TMatrix, double> precond_arg_pkg_dbl,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    Solve_Data<cascade::InnerOuterSolve, TMatrix> data;

    if (solver_id == "FP16") {

        cascade::PrecondArgPkg<TMatrix, __half> * precond_args_hlf_ptr = (
            precond_arg_pkg_dbl.cast_hlf_ptr()
        );
        cascade::TypedLinearSystem<TMatrix, __half> lin_sys_hlf(&gen_lin_sys);
        data = execute_solve<cascade::InnerOuterSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, __half>>(
                &lin_sys_hlf, u_hlf, solve_arg_pkg, *precond_args_hlf_ptr
            ),
            logger,
            false
        );
        delete precond_args_hlf_ptr;

    } else if (solver_id == "FP32") {

        cascade::PrecondArgPkg<TMatrix, float> * precond_args_sgl_ptr = (
            precond_arg_pkg_dbl.cast_sgl_ptr()
        );
        cascade::TypedLinearSystem<TMatrix, float> lin_sys_sgl(&gen_lin_sys);
        data = execute_solve<cascade::InnerOuterSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, float>>(
                &lin_sys_sgl, u_sgl, solve_arg_pkg, *precond_args_sgl_ptr
            ),
            logger,
            false
        );
        delete precond_args_sgl_ptr;

    } else if (solver_id == "FP64") {

        cascade::TypedLinearSystem<TMatrix, double> lin_sys_dbl(&gen_lin_sys);
        data = execute_solve<cascade::InnerOuterSolve, TMatrix>(
            solver_id,
            std::make_shared<cascade::FP_GMRES_IR_Solve<TMatrix, double>>(
                &lin_sys_dbl, u_dbl, solve_arg_pkg, precond_arg_pkg_dbl
            ),
            logger,
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
void run_record_vpgmres_solve(
    std::string solver_id,
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
    cascade::SolveArgPkg solve_arg_pkg,
    cascade::PrecondArgPkg<TMatrix, double> precond_arg_pkg_dbl,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    Solve_Data<cascade::VP_GMRES_IR_Solve, TMatrix> data;
    std::shared_ptr<cascade::VP_GMRES_IR_Solve<TMatrix>> solver_ptr;
    
    if (solver_id == "OuterRestartCount") {

        solver_ptr = (
            std::make_shared<cascade::OuterRestartCount<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );

    } else if (solver_id == "RelativeResidualThreshold") {

        solver_ptr = (
            std::make_shared<cascade::RelativeResidualThreshold<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );
    
    } else if (solver_id == "CheckStagnation") {

        solver_ptr = (
            std::make_shared<cascade::CheckStagnation<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );
    
    } else if (solver_id == "ProjectThresholdAfterStagnation") {

        solver_ptr = (
            std::make_shared<cascade::ProjectThresholdAfterStagnation<TMatrix>>(
                &gen_lin_sys, solve_arg_pkg, precond_arg_pkg_dbl
            )
        );

    } else {

        std::runtime_error(
            "run_record_vpgmres_solve: invalid mixed precision solver_id "
            "encountered"
        );

    }

    data = execute_solve<cascade::VP_GMRES_IR_Solve, TMatrix>(
        solver_id, solver_ptr, logger, false
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

        if (Solve_Group::valid_fp_solver_ids.count(solver_id) == 1) {

            run_record_fpgmres_solve(
                solver_id,
                gen_lin_sys,
                solve_group.solver_args,
                precond_data.precond_arg_pkg_dbl,
                output_data_dir,
                logger
            );

        } else if (Solve_Group::valid_vp_solver_ids.count(solver_id) == 1) {

            run_record_vpgmres_solve(
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

    fs::path solve_group_dir = output_data_dir / fs::path(solve_group.id);
    create_directory_if_nexists(solve_group_dir, outer_logger);

    Experiment_Log solve_group_logger(
        solve_group.id + "_logger",
        solve_group_dir / fs::path("log.log"),
        false
    );
    outer_logger.info("Start Solve_Group: " + solve_group.id);
    solve_group_logger.info("Start Solve_Group: " + solve_group.id);

    solve_group_logger.info(
        "Solve_Group solve args: " +
        solve_group.solver_args.get_info_string()
    );
    solve_group_logger.info(
        "Solve_Group precond args: " +
        solve_group.precond_specs.get_info_string()
    );

    solve_group.record_json(
        "solve_group_specs", solve_group_dir, solve_group_logger
    );

    // Iterate over matrices and iterations per matrix
    for (std::string matrix_file : solve_group.matrices_to_test) {

        fs::path matrix_output_data_dir = (
            solve_group_dir / fs::path(matrix_file).stem()
        );
        create_directory_if_nexists(matrix_output_data_dir, solve_group_logger);

        solve_group_logger.info(
            "Start matrix experimentation: " + matrix_file
        );
        
        int total_iters = solve_group.experiment_iterations;
        for (int exp_iter = 0; exp_iter < total_iters; ++exp_iter) {

            fs::path iter_output_data_dir = (
                matrix_output_data_dir / fs::path(std::to_string(exp_iter))
            );
            create_directory_if_nexists(
                iter_output_data_dir, solve_group_logger
            );

            solve_group_logger.info(
                "Start iteration: " + std::to_string(exp_iter)
            );

            Experiment_Log iter_logger(
                (solve_group.id + "_" + matrix_file + "_" +
                 std::to_string(exp_iter) + "_logger"),
                iter_output_data_dir / fs::path("log.log"),
                false
            );

            // Load linear system, generating b to solve
            cascade::GenericLinearSystem<TMatrix> gen_lin_sys = (
                load_lin_sys<TMatrix>(
                    cu_handles, matrix_data_dir, matrix_file, iter_logger
                )
            );

            run_record_solversuite_experiment(
                gen_lin_sys,
                solve_group,
                iter_output_data_dir,
                iter_logger
            );

            solve_group_logger.info(
                "Finish iteration: " + std::to_string(exp_iter)
            );

        }

        solve_group_logger.info(
            "Finish matrix experimentation: " + matrix_file
        );

    }

    outer_logger.info("Finish Solve_Group: " + solve_group.id);
    solve_group_logger.info("Finish Solve_Group: " + solve_group.id);

}

void run_record_experimental_spec(
    const cascade::cuHandleBundle &cu_handles,
    Experiment_Spec exp_spec,
    fs::path matrix_data_dir,
    fs::path output_data_dir,
    Experiment_Log logger
);

#endif
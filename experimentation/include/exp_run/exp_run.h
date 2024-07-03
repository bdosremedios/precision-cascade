#ifndef EXP_RUN_H
#define EXP_RUN_H

#include "exp_record/exp_record.h"
#include "exp_tools/exp_tools.h"

#include "tools/arg_pkgs/PrecondArgPkg.h"
#include "tools/arg_pkgs/LinearSystem.h"

#include <stdexcept>
#include <memory>
#include <string>

template <template <typename> typename TMatrix>
Preconditioner_Data<TMatrix> calc_preconditioner(
    const cascade::GenericLinearSystem<TMatrix> &gen_lin_sys,
    Preconditioner_Spec precond_specs,
    Experiment_Log logger
) {

    Experiment_Clock exp_clock;

    cascade::PrecondArgPkg<TMatrix, double> precond_args_dbl;

    logger.info("Start Precond Calc: " + precond_specs.get_spec_string());

    if (precond_specs.name == "none") {

        exp_clock.start_clock_experiment();
        precond_args_dbl = cascade::PrecondArgPkg<TMatrix, double>(
            std::make_shared<cascade::NoPreconditioner<TMatrix, double>>()
        );
        exp_clock.stop_clock_experiment();

    } else if (precond_specs.name == "jacobi") {

        exp_clock.start_clock_experiment();
        precond_args_dbl = cascade::PrecondArgPkg<TMatrix, double>(
            std::make_shared<cascade::JacobiPreconditioner<TMatrix, double>>(
                gen_lin_sys.get_A()
            )
        );
        exp_clock.stop_clock_experiment();

    } else if (precond_specs.name == "ilu0") {

        exp_clock.start_clock_experiment();
        std::shared_ptr<cascade::ILUPreconditioner<TMatrix, double>> ilu0 = (
            std::make_shared<cascade::ILUPreconditioner<TMatrix, double>>(
                gen_lin_sys.get_A()
            )
        );
        exp_clock.stop_clock_experiment();

        logger.info("Precond: L info: " + ilu0->get_L().get_info_string());
        logger.info("Precond: U info: " + ilu0->get_U().get_info_string());
        precond_args_dbl = cascade::PrecondArgPkg<TMatrix, double>(ilu0);

    } else if (precond_specs.name == "ilutp") {

        exp_clock.start_clock_experiment();
        std::shared_ptr<cascade::ILUPreconditioner<TMatrix, double>> ilutp = (
            std::make_shared<cascade::ILUPreconditioner<TMatrix, double>>(
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
        precond_args_dbl = cascade::PrecondArgPkg<TMatrix, double>(ilutp);

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

#endif
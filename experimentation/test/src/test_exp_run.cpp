#include "test_experiment.h"

#include "exp_run/exp_run_record.h"

class Test_Experiment_Run: public Test_Experiment_Base
{
private:

    std::string get_type_str(std::string solver_id) {
        if (solver_id == "FP16") {
            return "__half";
        } else if (solver_id == "FP32") {
            #ifdef WIN32
            return "float";
            #else
            return "f";
            #endif
        } else {
            #ifdef WIN32
            return "double";
            #else
            return "d";
            #endif
        }
    }

    std::string get_mat_type_str(std::string mat_type_id) {
        if (mat_type_id == "dense") {
            return "MatrixDense";
        } else if (mat_type_id == "sparse") {
            return "NoFillMatrixSparse";
        } else {
            throw std::runtime_error(
                "get_mat_type_str: no matching mat_type_id"
            );
        }
    }

    std::string get_solver_str(std::string solver_id) {

        if (
            (solver_id == "FP16") ||
            (solver_id == "FP32") ||
            (solver_id == "FP64")
        ) {
            return "FP_GMRES_IR_Solve";
        } else if (solver_id == "RelativeResidualThreshold") {
            return "RelativeResidualThreshold";
        } else if (solver_id == "OuterRestartCount") {
            return "OuterRestartCount";
        } else if (solver_id == "CheckStagnation") {
            return "CheckStagnation";
        } else if (solver_id == "ThresholdToStagnation") {
            return "ThresholdToStagnation";
        } else if (solver_id == "SD_RelativeResidualThreshold") {
            return "SD_RelativeResidualThreshold";
        } else if (solver_id == "SD_OuterRestartCount") {
            return "SD_OuterRestartCount";
        } else if (solver_id == "SD_CheckStagnation") {
            return "SD_CheckStagnation";
        } else {
            throw std::runtime_error("get_solver_str: no matching solver_id");
        }

    }

    std::string get_left_precond(
        std::string solver_id,
        Preconditioner_Spec precond_specs
    ) {

        std::string type_str = get_type_str(solver_id);

        if (precond_specs.name == "none") {
            return "NoPreconditioner";
        } else if (precond_specs.name == "jacobi") {
            if ((solver_id == "FP16") || (solver_id == "FP32")) {
                return "MatrixInversePreconditioner";
            } else {
                return "JacobiPreconditioner";
            }
        } else if (
            (precond_specs.name == "ilu0") ||
            (precond_specs.name == "ilutp")
        ) {
            return "ILUPreconditioner";
        } else {
            throw std::runtime_error(
                "get_left_precond: no matching precond_specs"
            );
        }

    }

    std::string get_right_precond(
        std::string solver_id,
        Preconditioner_Spec precond_specs
    ) {
        return "NoPreconditioner";

    }

    bool contains_strings(
        std::string main_str, std::vector<std::string> to_match
    ) {
        for (std::string match_str: to_match) {
            if (main_str.find(match_str) == std::string::npos) {
                return false;
            }
        }
        return true;
    }

public:

    Experiment_Log logger;

    Test_Experiment_Run() {
        logger = Experiment_Log();
    }

    ~Test_Experiment_Run() {}

    template <template <typename> typename TMatrix>
    void Test_Calc_Preconditioner() {

        GenericLinearSystem<TMatrix> gen_lin_sys(
            TMatrix<double>(MatrixDense<double>::Random(
                *Test_Experiment_Base::cu_handles_ptr, 16, 16
            )),
            Vector<double>::Random(*Test_Experiment_Base::cu_handles_ptr, 16)
        );

        Preconditioner_Spec precond_spec_none("none");
        Preconditioner_Data<TMatrix> none_data = calc_preconditioner<TMatrix>(
            gen_lin_sys, precond_spec_none, logger
        );
        ASSERT_EQ(none_data.id, precond_spec_none.get_spec_string());
        ASSERT_TRUE(none_data.clock.check_completed());
        ASSERT_EQ(
            typeid(*none_data.precond_arg_pkg_dbl.left_precond).name(),
            typeid(NoPreconditioner<TMatrix, double>).name()
        );
        ASSERT_EQ(
            typeid(*none_data.precond_arg_pkg_dbl.right_precond).name(),
            typeid(NoPreconditioner<TMatrix, double>).name()
        );

        Preconditioner_Spec precond_spec_jacobi("jacobi");
        Preconditioner_Data<TMatrix> jacobi_data = calc_preconditioner<TMatrix>(
            gen_lin_sys, precond_spec_jacobi, logger
        );
        ASSERT_EQ(jacobi_data.id, precond_spec_jacobi.get_spec_string());
        ASSERT_TRUE(jacobi_data.clock.check_completed());
        JacobiPreconditioner<TMatrix, double> jacobi_precond(
            gen_lin_sys.get_A()
        );
        ASSERT_EQ(
            typeid(*jacobi_data.precond_arg_pkg_dbl.left_precond).name(),
            typeid(jacobi_precond).name()
        );
        ASSERT_VECTOR_NEAR(
            jacobi_data.precond_arg_pkg_dbl.left_precond->action_inv_M(
                gen_lin_sys.get_b()
            ),
            jacobi_precond.action_inv_M(gen_lin_sys.get_b()),
            u_dbl
        );
        ASSERT_EQ(
            typeid(*jacobi_data.precond_arg_pkg_dbl.right_precond).name(),
            typeid(NoPreconditioner<TMatrix, double>).name()
        );

        Preconditioner_Spec precond_spec_ilu0("ilu0");
        Preconditioner_Data<TMatrix> ilu0_data = calc_preconditioner<TMatrix>(
            gen_lin_sys, precond_spec_ilu0, logger
        );
        ASSERT_EQ(ilu0_data.id, precond_spec_ilu0.get_spec_string());
        ASSERT_TRUE(ilu0_data.clock.check_completed());
        ILUPreconditioner<TMatrix, double> ilu0_precond(gen_lin_sys.get_A());
        ASSERT_EQ(
            typeid(*ilu0_data.precond_arg_pkg_dbl.left_precond).name(),
            typeid(ilu0_precond).name()
        );
        ASSERT_VECTOR_NEAR(
            ilu0_data.precond_arg_pkg_dbl.left_precond->action_inv_M(
                gen_lin_sys.get_b()
            ),
            ilu0_precond.action_inv_M(gen_lin_sys.get_b()),
            u_dbl
        );
        ASSERT_EQ(
            typeid(*ilu0_data.precond_arg_pkg_dbl.right_precond).name(),
            typeid(NoPreconditioner<TMatrix, double>).name()
        );

        Preconditioner_Spec precond_spec_ilutp("ilutp", 0.1, 10);
        Preconditioner_Data<TMatrix> ilutp_data = calc_preconditioner<TMatrix>(
            gen_lin_sys, precond_spec_ilutp, logger
        );
        ASSERT_EQ(ilutp_data.id, precond_spec_ilutp.get_spec_string());
        ASSERT_TRUE(ilutp_data.clock.check_completed());
        ILUPreconditioner<TMatrix, double> ilutp_precond(
            gen_lin_sys.get_A(), 0.1, 10, true
        );
        ASSERT_EQ(
            typeid(*ilutp_data.precond_arg_pkg_dbl.left_precond).name(),
            typeid(ilutp_precond).name()
        );
        ASSERT_VECTOR_NEAR(
            ilutp_data.precond_arg_pkg_dbl.left_precond->action_inv_M(
                gen_lin_sys.get_b()
            ),
            ilutp_precond.action_inv_M(gen_lin_sys.get_b()),
            u_dbl
        );
        ASSERT_EQ(
            typeid(*ilutp_data.precond_arg_pkg_dbl.right_precond).name(),
            typeid(NoPreconditioner<TMatrix, double>).name()
        );

    }

    template <template <typename> typename TMatrix>
    void Test_Execute_FP_GMRES_IR_Solve() {

        GenericLinearSystem<TMatrix> gen_lin_sys(
            TMatrix<double>(MatrixDense<double>::Random(
                *Test_Experiment_Base::cu_handles_ptr, 16, 16
            )),
            Vector<double>::Random(*Test_Experiment_Base::cu_handles_ptr, 16)
        );
        SolveArgPkg solver_args(10, 10, 0.05);

        TypedLinearSystem<TMatrix, __half> typed_lin_sys_hlf(&gen_lin_sys);
        std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, __half>> fpgmres16_ptr(
            std::make_shared<FP_GMRES_IR_Solve<TMatrix, __half>>(
                &typed_lin_sys_hlf, u_hlf, solver_args
            )
        );
        ASSERT_FALSE(fpgmres16_ptr->check_initiated());
        ASSERT_FALSE(fpgmres16_ptr->check_terminated());

        Experiment_Clock fpgmres16_clock;
        fpgmres16_clock.start_clock_experiment();
        Solve_Data<InnerOuterSolve, TMatrix> fpgmres16_solve_data(
            execute_solve<InnerOuterSolve, TMatrix>(
                "fp16_id", fpgmres16_ptr, logger, false
            )
        );
        fpgmres16_clock.stop_clock_experiment();

        ASSERT_EQ(fpgmres16_solve_data.id, "fp16_id");
        ASSERT_TRUE(fpgmres16_solve_data.clock.check_completed());
        ASSERT_NEAR(
            fpgmres16_clock.get_elapsed_time_ms(),
            fpgmres16_solve_data.clock.get_elapsed_time_ms(),
            1
        );
        ASSERT_EQ(fpgmres16_ptr, fpgmres16_solve_data.solver_ptr);
        ASSERT_TRUE(fpgmres16_solve_data.solver_ptr->check_initiated());
        ASSERT_TRUE(fpgmres16_solve_data.solver_ptr->check_terminated());

        TypedLinearSystem<TMatrix, float> typed_lin_sys_sgl(&gen_lin_sys);
        std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, float>> fpgmres32_ptr(
            std::make_shared<FP_GMRES_IR_Solve<TMatrix, float>>(
                &typed_lin_sys_sgl, u_sgl, solver_args
            )
        );
        ASSERT_FALSE(fpgmres32_ptr->check_initiated());
        ASSERT_FALSE(fpgmres32_ptr->check_terminated());

        Experiment_Clock fpgmres32_clock;
        fpgmres32_clock.start_clock_experiment();
        Solve_Data<InnerOuterSolve, TMatrix> fpgmres32_solve_data(
            execute_solve<InnerOuterSolve, TMatrix>(
                "fp32_id", fpgmres32_ptr, logger, false
            )
        );
        fpgmres32_clock.stop_clock_experiment();

        ASSERT_EQ(fpgmres32_solve_data.id, "fp32_id");
        ASSERT_TRUE(fpgmres32_solve_data.clock.check_completed());
        ASSERT_NEAR(
            fpgmres32_clock.get_elapsed_time_ms(),
            fpgmres32_solve_data.clock.get_elapsed_time_ms(),
            1
        );
        ASSERT_EQ(fpgmres32_ptr, fpgmres32_solve_data.solver_ptr);
        ASSERT_TRUE(fpgmres32_solve_data.solver_ptr->check_initiated());
        ASSERT_TRUE(fpgmres32_solve_data.solver_ptr->check_terminated());

        TypedLinearSystem<TMatrix, double> typed_lin_sys_dbl(&gen_lin_sys);
        std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, double>> fpgmres64_ptr(
            std::make_shared<FP_GMRES_IR_Solve<TMatrix, double>>(
                &typed_lin_sys_dbl, u_dbl, solver_args
            )
        );
        ASSERT_FALSE(fpgmres64_ptr->check_initiated());
        ASSERT_FALSE(fpgmres64_ptr->check_terminated());

        Experiment_Clock fpgmres64_clock;
        fpgmres64_clock.start_clock_experiment();
        Solve_Data<InnerOuterSolve, TMatrix> fpgmres64_solve_data(
            execute_solve<InnerOuterSolve, TMatrix>(
                "fp64_id", fpgmres64_ptr, logger, false
            )
        );
        fpgmres64_clock.stop_clock_experiment();

        ASSERT_EQ(fpgmres64_solve_data.id, "fp64_id");
        ASSERT_TRUE(fpgmres64_solve_data.clock.check_completed());
        ASSERT_NEAR(
            fpgmres64_clock.get_elapsed_time_ms(),
            fpgmres64_solve_data.clock.get_elapsed_time_ms(),
            1
        );
        ASSERT_EQ(fpgmres64_ptr, fpgmres64_solve_data.solver_ptr);
        ASSERT_TRUE(fpgmres64_solve_data.solver_ptr->check_initiated());
        ASSERT_TRUE(fpgmres64_solve_data.solver_ptr->check_terminated());

    }

    template <template <typename> typename TMatrix>
    void Test_Execute_VP_GMRES_IR_Solve() {

        GenericLinearSystem<TMatrix> gen_lin_sys(
            TMatrix<double>(MatrixDense<double>::Random(
                *Test_Experiment_Base::cu_handles_ptr, 16, 16
            )),
            Vector<double>::Random(*Test_Experiment_Base::cu_handles_ptr, 16)
        );
        SolveArgPkg solver_args(10, 10, 0.05);

        std::shared_ptr<RelativeResidualThreshold<TMatrix>> rrt_vpgmres_ptr(
            std::make_shared<RelativeResidualThreshold<TMatrix>>(
                &gen_lin_sys, solver_args
            )
        );
        ASSERT_FALSE(rrt_vpgmres_ptr->check_initiated());
        ASSERT_FALSE(rrt_vpgmres_ptr->check_terminated());

        Experiment_Clock rrt_vpgmres_clock;
        rrt_vpgmres_clock.start_clock_experiment();
        Solve_Data<InnerOuterSolve, TMatrix> rrt_vpgmres_solve_data(
            execute_solve<InnerOuterSolve, TMatrix>(
                "id_rrt_vpgmres", rrt_vpgmres_ptr, logger, false
            )
        );
        rrt_vpgmres_clock.stop_clock_experiment();

        ASSERT_EQ(rrt_vpgmres_solve_data.id, "id_rrt_vpgmres");
        ASSERT_TRUE(rrt_vpgmres_solve_data.clock.check_completed());
        ASSERT_NEAR(
            rrt_vpgmres_clock.get_elapsed_time_ms(),
            rrt_vpgmres_solve_data.clock.get_elapsed_time_ms(),
            1
        );
        ASSERT_EQ(rrt_vpgmres_ptr, rrt_vpgmres_solve_data.solver_ptr);
        ASSERT_TRUE(rrt_vpgmres_solve_data.solver_ptr->check_initiated());
        ASSERT_TRUE(rrt_vpgmres_solve_data.solver_ptr->check_terminated());

        std::shared_ptr<OuterRestartCount<TMatrix>> orc_vpgmres_ptr(
            std::make_shared<OuterRestartCount<TMatrix>>(
                &gen_lin_sys, solver_args
            )
        );
        ASSERT_FALSE(orc_vpgmres_ptr->check_initiated());
        ASSERT_FALSE(orc_vpgmres_ptr->check_terminated());

        Experiment_Clock orc_vpgmres_clock;
        orc_vpgmres_clock.start_clock_experiment();
        Solve_Data<InnerOuterSolve, TMatrix> orc_vpgmres_solve_data(
            execute_solve<InnerOuterSolve, TMatrix>(
                "id_orc_vpgmres", orc_vpgmres_ptr, logger, false
            )
        );
        orc_vpgmres_clock.stop_clock_experiment();

        ASSERT_EQ(orc_vpgmres_solve_data.id, "id_orc_vpgmres");
        ASSERT_TRUE(orc_vpgmres_solve_data.clock.check_completed());
        ASSERT_NEAR(
            orc_vpgmres_clock.get_elapsed_time_ms(),
            orc_vpgmres_solve_data.clock.get_elapsed_time_ms(),
            1
        );
        ASSERT_EQ(orc_vpgmres_ptr, orc_vpgmres_solve_data.solver_ptr);
        ASSERT_TRUE(orc_vpgmres_solve_data.solver_ptr->check_initiated());
        ASSERT_TRUE(orc_vpgmres_solve_data.solver_ptr->check_terminated());

    }

    template <template <typename> typename TMatrix>
    void Test_Run_Record_Solve_Group(Solve_Group solve_group) {

        run_record_solve_group<TMatrix>(
            *Test_Experiment_Base::cu_handles_ptr,
            solve_group,
            test_data_dir,
            test_output_dir,
            logger
        );

        fs::path solve_group_dir(test_output_dir / fs::path(solve_group.id));

        // Test that the right thing was run in each case
        for (std::string matrix_str : solve_group.matrices_to_test) {
            for (std::string solver_id : solve_group.solvers_to_use) {
                for (
                    int exp_iter=0;
                    exp_iter < solve_group.experiment_iterations;
                    ++exp_iter
                ) {

                    fs::path file_path(
                        solve_group_dir /
                        fs::path(matrix_str).stem() /
                        fs::path(std::to_string(exp_iter)) /
                        fs::path(solver_id + ".json")
                    );
                    std::ifstream file_in(file_path);

                    if (!(file_in.is_open())) {
                        throw std::runtime_error(
                            "Failed to read "+file_path.string()
                        );
                    }

                    json loaded_file = json::parse(file_in);

                    ASSERT_EQ(loaded_file["id"], solver_id);
                    if (solver_id.find("FP") != std::string::npos) {
                        ASSERT_TRUE(
                            contains_strings(
                                loaded_file["solver_class"],
                                std::vector<std::string>(
                                    {get_type_str(solver_id),
                                     get_mat_type_str(solve_group.matrix_type),
                                     get_solver_str(solver_id)}
                                )
                            )
                        );
                    } else {
                        ASSERT_TRUE(
                            contains_strings(
                                loaded_file["solver_class"],
                                std::vector<std::string>(
                                    {get_mat_type_str(solve_group.matrix_type),
                                     get_solver_str(solver_id)}
                                )
                            )
                        );
                    }
            
                    ASSERT_EQ(loaded_file["initiated"], "true");
                    ASSERT_EQ(loaded_file["terminated"], "true");

                }
            }
        }

    }

    template <template <typename> typename TMatrix>
    void Test_Run_Record_Experiment_Spec(Experiment_Spec exp_spec) {

        run_record_experimental_spec(
            *Test_Experiment_Base::cu_handles_ptr,
            exp_spec,
            test_data_dir,
            test_output_dir,
            logger
        );

    }

};

TEST_F(Test_Experiment_Run, Test_Calc_Preconditioner) {
    Test_Calc_Preconditioner<MatrixDense>();
    Test_Calc_Preconditioner<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Run, Test_Execute_FP_GMRES_IR_Solve) {
    Test_Execute_FP_GMRES_IR_Solve<MatrixDense>();
    Test_Execute_FP_GMRES_IR_Solve<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Run, Test_Execute_VP_GMRES_IR_Solve) {
    Test_Execute_VP_GMRES_IR_Solve<MatrixDense>();
    Test_Execute_VP_GMRES_IR_Solve<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Run, Test_AllSolvers_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "allsolvers_dense",
        std::vector<std::string>(
            {"FP16", "FP32", "FP64",
             "RelativeResidualThreshold",
             "OuterRestartCount",
             "CheckStagnation",
             "ThresholdToStagnation",
             "SD_RelativeResidualThreshold",
             "SD_OuterRestartCount",
             "SD_CheckStagnation",}
        ),
        "dense", 3, 10, 4, 1e-4,
        Preconditioner_Spec("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Record_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "allsolvers_sparse",
        std::vector<std::string>(
            {"FP16", "FP32", "FP64",
             "RelativeResidualThreshold",
             "OuterRestartCount",
             "CheckStagnation",
             "ThresholdToStagnation",
             "SD_RelativeResidualThreshold",
             "SD_OuterRestartCount",
             "SD_CheckStagnation",}
        ),
        "sparse", 3, 10, 4, 1e-4,
        Preconditioner_Spec("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Record_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

}

TEST_F(Test_Experiment_Run, Test_AllPreconditioners_Run_Solve_Group) {

    std::vector<Preconditioner_Spec> precond_spec_vec {
        Preconditioner_Spec("none"),
        Preconditioner_Spec("jacobi"),
        Preconditioner_Spec("ilu0"),
        Preconditioner_Spec("ilutp", 1e-4, 20)
    };

    for (Preconditioner_Spec precond_specs : precond_spec_vec) {

        Solve_Group solve_group_dense(
            "allpreconditioners_dense_"+precond_specs.name,
            std::vector<std::string>(
                {"FP16", "FP32", "FP64", "OuterRestartCount"}
            ),
            "dense", 3, 10, 4, 1e-4,
            precond_specs,
            std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
        );

        Test_Run_Record_Solve_Group<MatrixDense>(solve_group_dense);

        Solve_Group solve_group_sparse(
            "allpreconditioners_sparse_"+precond_specs.name,
            std::vector<std::string>(
                {"FP16", "FP32", "FP64", "OuterRestartCount"}
            ),
            "sparse", 3, 10, 4, 1e-4,
            precond_specs,
            std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
        );

        Test_Run_Record_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

    }

}

TEST_F(Test_Experiment_Run, Test_Mix_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "mixsolvers_dense",
        std::vector<std::string>(
            {"FP64", "FP16", "RelativeResidualThreshold"}
        ),
        "dense", 3, 10, 4, 1e-4,
        Preconditioner_Spec("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Record_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "mixsolvers_sparse",
        std::vector<std::string>(
            {"FP64", "FP16", "RelativeResidualThreshold"}
        ),
        "sparse", 3, 10, 4, 1e-4,
        Preconditioner_Spec("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Record_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

}
#include "test_experiment.h"

#include "experiment_run.h"

class TestRun: public TestExperimentBase
{
public:

    Experiment_Log logger;

    TestRun() {
        logger = Experiment_Log();
    }

    ~TestRun() {}

    template <template <typename> typename M>
    void Test_Run_Solve_Group(Solve_Group solve_group) {

        run_solve_group<M>(
            *TestExperimentBase::cu_handles_ptr, solve_group, test_data_dir, test_output_dir, logger
        );

        fs::path solve_group_dir(test_output_dir / fs::path(solve_group.id));

        // for (std::string matrix_str : solve_group.matrices_to_test) {
        //     for (std::string solver_id : solve_group.solvers_to_use) {
        //         for (int exp_iter=0; exp_iter<solve_group.experiment_iterations; ++exp_iter) {

        //             std::string experiment_id(
        //                 std::format(
        //                     "{}_{}_{}", matrix_str, solver_id, exp_iter
        //                 )
        //             );
        //             fs::path file_path = solve_group_dir / fs::path(experiment_id+".json");
        //             std::ifstream file_in(file_path);
        //             json loaded_file = json::parse(file_in);

        //             // ASSERT_EQ(loaded_file["solver_class"], typeid(*solve_ptr).name());
        //             // ASSERT_EQ(loaded_file["initiated"], "true");
        //             // ASSERT_EQ(loaded_file["converged"], bool_to_string(solve_ptr->check_converged()));
        //             // ASSERT_EQ(loaded_file["terminated"], "true");
        //             // ASSERT_EQ(loaded_file["iteration"], solve_ptr->get_iteration());
        //             // ASSERT_EQ(loaded_file["elapsed_time_ms"], data.clock.get_elapsed_time_ms());

        //         }
        //     }
        // }

    }

    template <template <typename> typename M>
    void Test_Run_Experiment_Specification(Experiment_Specification exp_spec) {

        run_experimental_spec(
            *TestExperimentBase::cu_handles_ptr, exp_spec, test_data_dir, test_output_dir, logger
        );

    }

};

TEST_F(TestRun, Test_All_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "name1_dense",
        std::vector<std::string>({"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}),
        "dense", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4", "easy_5_5"})
    );

    Test_Run_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "name1_sparse",
        std::vector<std::string>({"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}),
        "sparse", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4", "easy_5_5"})
    );

    Test_Run_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

}

TEST_F(TestRun, Test_Some_Run_Solve_Group) {

}
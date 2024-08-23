#include "test_experiment.h"

#include "exp_spec/exp_spec.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Test_Experiment_Spec: public Test_Experiment_Base
{
public:

    Experiment_Log logger;

    void ASSERT_SOLVE_GROUP_JSON_MATCH(fs::path json_file, Solve_Group sgroup) {

        std::ifstream file_in(json_file);
        if (!file_in.is_open()) { FAIL(); }

        json loaded_file = json::parse(file_in);

        ASSERT_EQ(loaded_file["id"], sgroup.id);
        ASSERT_EQ(loaded_file["matrix_type"], sgroup.matrix_type);

        ASSERT_EQ(
            loaded_file["experiment_iterations"],
            sgroup.experiment_iterations
        );
        ASSERT_EQ(
            loaded_file["max_outer_iterations"],
            sgroup.solver_args.max_iter
        );
        ASSERT_EQ(
            loaded_file["max_inner_iterations"],
            sgroup.solver_args.max_inner_iter
        );

        ASSERT_EQ(
            loaded_file["target_rel_res"],
            sgroup.solver_args.target_rel_res
        );

        ASSERT_EQ(
            loaded_file["precond_specs"],
            sgroup.precond_specs.get_spec_string()
        );

        ASSERT_EQ(
            loaded_file["solver_ids"].size(),
            sgroup.solvers_to_use.size()
        );
        for (int i=0; i<sgroup.solvers_to_use.size(); ++i) {
            ASSERT_EQ(
                loaded_file["solver_ids"][i],
                sgroup.solvers_to_use[i]
            );
        }

        ASSERT_EQ(
            loaded_file["matrix_ids"].size(),
            sgroup.matrices_to_test.size()
        );
        for (int i=0; i<sgroup.matrices_to_test.size(); ++i) {

            std::string matrix_id = (
                fs::path(sgroup.matrices_to_test[i]).stem().string()
            );
            
            ASSERT_EQ(loaded_file["matrix_ids"][i], matrix_id);

        }


        file_in.close();
        
    }

};

TEST_F(Test_Experiment_Spec, Test_Preconditioner_Spec_Basic_Construct) {

    Preconditioner_Spec precond_spec_none("none");
    ASSERT_EQ(precond_spec_none.name, "none");

    Preconditioner_Spec precond_spec_jacobi("jacobi");
    ASSERT_EQ(precond_spec_jacobi.name, "jacobi");

    Preconditioner_Spec precond_spec_ilu0("ilu0");
    ASSERT_EQ(precond_spec_ilu0.name, "ilu0");

    Preconditioner_Spec precond_spec_ilutp(
        "ilutp", 2e-3, 43
    );
    ASSERT_EQ(precond_spec_ilutp.name, "ilutp");
    ASSERT_EQ(precond_spec_ilutp.ilutp_tau, 2e-3);
    ASSERT_EQ(precond_spec_ilutp.ilutp_p, 43);

    Preconditioner_Spec precond_spec_ilutp_2(
        "ilutp", 0., 2
    );
    ASSERT_EQ(precond_spec_ilutp_2.name, "ilutp");
    ASSERT_EQ(precond_spec_ilutp_2.ilutp_tau, 0.);
    ASSERT_EQ(precond_spec_ilutp_2.ilutp_p, 2);

}

TEST_F(Test_Experiment_Spec, Test_Preconditioner_Spec_Bad) {

    // Bad name string
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() { Preconditioner_Spec precond_spec("asdf"); }
    );

    // Bad 3 array name string
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() { Preconditioner_Spec precond_spec("asdf", 2e-3, 43); }
    );

    // Bad 3 array name of correct name but not valid for 3
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() { Preconditioner_Spec precond_spec("ilu0", 2e-3, 43); }
    );

    // Bad tau tolerance
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() { Preconditioner_Spec precond_spec("ilutp", -1., 43); }
    );

    // Bad p number of non-zeros
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() { Preconditioner_Spec precond_spec("ilutp", 2e-3, 0); }
    );

}

TEST_F(Test_Experiment_Spec, Test_Solve_Group_Basic_Construct) {

    Solve_Group solve_group(
        "id1",
        std::vector<std::string>({"FP32", "RelativeResidualThreshold"}),
        "dense",
        15,
        23,
        101,
        3e-6,
        Preconditioner_Spec("ilu0"),
        std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
    );

    ASSERT_EQ(solve_group.id, "id1");
    ASSERT_EQ(
        solve_group.solvers_to_use,
        std::vector<std::string>({"FP32", "RelativeResidualThreshold"})
    );
    ASSERT_EQ(solve_group.matrix_type, "dense");
    ASSERT_EQ(solve_group.experiment_iterations, 15);
    ASSERT_EQ(solve_group.solver_args.max_iter, 23);
    ASSERT_EQ(solve_group.solver_args.max_inner_iter, 101);
    ASSERT_EQ(solve_group.solver_args.target_rel_res, 3e-6);
    ASSERT_EQ(solve_group.precond_specs, Preconditioner_Spec("ilu0"));
    ASSERT_EQ(
        solve_group.matrices_to_test,
        std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
    );

    Solve_Group solve_group_2(
        "adsfhjl",
        std::vector<std::string>({"OuterRestartCount", "FP16", "FP64"}),
        "sparse",
        12,
        5,
        2,
        0.,
        Preconditioner_Spec("ilutp", 2.3e-3, 12),
        std::vector<std::string>({"asdfasdfga.mtx"})
    );

    ASSERT_EQ(solve_group_2.id, "adsfhjl");
    ASSERT_EQ(
        solve_group_2.solvers_to_use,
        std::vector<std::string>({"OuterRestartCount", "FP16", "FP64"})
    );
    ASSERT_EQ(solve_group_2.matrix_type, "sparse");
    ASSERT_EQ(solve_group_2.experiment_iterations, 12);
    ASSERT_EQ(solve_group_2.solver_args.max_iter, 5);
    ASSERT_EQ(solve_group_2.solver_args.max_inner_iter, 2);
    ASSERT_EQ(solve_group_2.solver_args.target_rel_res, 0);
    ASSERT_EQ(
        solve_group_2.precond_specs,
        Preconditioner_Spec("ilutp", 2.3e-3, 12)
    );
    ASSERT_EQ(
        solve_group_2.matrices_to_test,
        std::vector<std::string>({"asdfasdfga.mtx"})
    );

}

TEST_F(Test_Experiment_Spec, Test_Solve_Group_Write_Json) {

    Solve_Group solve_group_1(
        "id1",
        std::vector<std::string>({"FP32", "RelativeResidualThreshold"}),
        "dense",
        15,
        23,
        101,
        3e-6,
        Preconditioner_Spec("ilu0"),
        std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
    );

    solve_group_1.record_json("solve_group_1", test_output_dir, logger);

    ASSERT_SOLVE_GROUP_JSON_MATCH(
        test_output_dir / fs::path("solve_group_1.json"),
        solve_group_1
    );

    Solve_Group solve_group_2(
        "adsfhjl",
        std::vector<std::string>({"OuterRestartCount", "FP16", "FP64"}),
        "sparse",
        12,
        5,
        2,
        0.,
        Preconditioner_Spec("ilutp", 2.3e-3, 12),
        std::vector<std::string>({"asdfasdfga.mtx"})
    );

    solve_group_2.record_json("solve_group_2", test_output_dir, logger);

    ASSERT_SOLVE_GROUP_JSON_MATCH(
        test_output_dir / fs::path("solve_group_2.json"),
        solve_group_2
    );

}

TEST_F(Test_Experiment_Spec, Test_Solve_Group_Bad) {

    // Bad solver in list
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold", "bad"}
                ),
                "dense",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );

    // Repeated solver in list
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold", "FP32"}
                ),
                "dense",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );
    
    // Bad matrix type
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "bad",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );

    // Bad experimentation iterations
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                0,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );

    // Bad solver max outer iterations
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                0,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );
    
    // Bad solver max inner iterations
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                23,
                0,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );
    
    // Bad solver target rel res
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                23,
                101,
                -3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.csv"})
            );
        }
    );

    // Matrices to test empty
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>()
            );
        }
    );

    // Matrices to test bad extension
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd.mtx", "asd.asdf"})
            );
        }
    );

    // Matrices to test missing extension
    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            Solve_Group solve_group(
                "id1",
                std::vector<std::string>(
                    {"FP32", "RelativeResidualThreshold"}
                ),
                "dense",
                15,
                23,
                101,
                3e-6,
                Preconditioner_Spec("ilu0"),
                std::vector<std::string>({"asdfasd", "asd.csv"})
            );
        }
    );

}
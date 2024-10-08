#include "test_experiment.h"

#include "exp_read/exp_read.h"

class Test_Experiment_Read: public Test_Experiment_Base {};

TEST_F(Test_Experiment_Read, TestCorrectSingleEntryJson) {

    Experiment_Spec test_spec = parse_experiment_spec(
        test_json_dir / fs::path("good_spec_single.json")
    );

    ASSERT_EQ(test_spec.id, "good_spec_single");
    ASSERT_EQ(test_spec.solve_groups.size(), 1);
    ASSERT_EQ(test_spec.solve_groups[0].id, "solve_group_1");
    ASSERT_EQ(test_spec.solve_groups[0].experiment_iterations, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[0], "FP16");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[1], "FP32");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[2], "FP64");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[3], "OuterRestartCount");
    ASSERT_EQ(test_spec.solve_groups[0].matrix_type, "dense");
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_iter, 10);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_inner_iter, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.target_rel_res, 1e-10);
    ASSERT_EQ(
        test_spec.solve_groups[0].precond_specs,
        Preconditioner_Spec("none")
    );
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test.size(), 3);
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[0], "494_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[1], "662_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[2], "685_bus.csv");

}

TEST_F(Test_Experiment_Read, TestCorrectILUTPJson) {

    Experiment_Spec test_spec = parse_experiment_spec(
        test_json_dir / fs::path("good_spec_ILUTP.json")
    );

    ASSERT_EQ(test_spec.id, "good_spec_ILUTP");
    ASSERT_EQ(test_spec.solve_groups.size(), 1);
    ASSERT_EQ(test_spec.solve_groups[0].id, "solve_group_1");
    ASSERT_EQ(test_spec.solve_groups[0].experiment_iterations, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[0], "FP16");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[1], "FP32");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[2], "FP64");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[3], "OuterRestartCount");
    ASSERT_EQ(test_spec.solve_groups[0].matrix_type, "dense");
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_iter, 10);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_inner_iter, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.target_rel_res, 1e-10);
    ASSERT_EQ(
        test_spec.solve_groups[0].precond_specs,
        Preconditioner_Spec("ilutp", 1e-6, 21)
    );
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test.size(), 3);
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[0], "494_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[1], "662_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[2], "685_bus.csv");

}

TEST_F(Test_Experiment_Read, TestCorrectMultipleEntryJson) {

    Experiment_Spec test_spec = parse_experiment_spec(
        test_json_dir / fs::path("good_spec_multi.json")
    );

    ASSERT_EQ(test_spec.id, "good_spec_multi");
    ASSERT_EQ(test_spec.solve_groups.size(), 3);

    ASSERT_EQ(test_spec.solve_groups[0].id, "a");
    ASSERT_EQ(test_spec.solve_groups[0].experiment_iterations, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[0], "FP16");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[1], "FP32");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[2], "FP64");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[3], "OuterRestartCount");
    ASSERT_EQ(test_spec.solve_groups[0].matrix_type, "dense");
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_iter, 10);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.max_inner_iter, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solver_args.target_rel_res, 1e-10);
    ASSERT_EQ(
        test_spec.solve_groups[0].precond_specs,
        Preconditioner_Spec("none")
    );
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test.size(), 3);
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[0], "494_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[1], "662_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[0].matrices_to_test[2], "685_bus.csv");

    ASSERT_EQ(test_spec.solve_groups[1].id, "b");
    ASSERT_EQ(test_spec.solve_groups[1].experiment_iterations, 1);
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[0], "FP16");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[1], "FP32");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[2], "FP64");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[3], "OuterRestartCount");
    ASSERT_EQ(test_spec.solve_groups[1].matrix_type, "sparse");
    ASSERT_EQ(test_spec.solve_groups[1].solver_args.max_iter, 4);
    ASSERT_EQ(test_spec.solve_groups[1].solver_args.max_inner_iter, 4);
    ASSERT_EQ(test_spec.solve_groups[1].solver_args.target_rel_res, 3.5);
    ASSERT_EQ(
        test_spec.solve_groups[1].precond_specs,
        Preconditioner_Spec("ilu0")
    );
    ASSERT_EQ(test_spec.solve_groups[1].matrices_to_test.size(), 1);
    ASSERT_EQ(test_spec.solve_groups[1].matrices_to_test[0], "494_bus.csv");

    ASSERT_EQ(test_spec.solve_groups[2].id, "c");
    ASSERT_EQ(test_spec.solve_groups[2].experiment_iterations, 3);
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[2], "FP64");
    ASSERT_EQ(test_spec.solve_groups[0].solvers_to_use[3], "OuterRestartCount");
    ASSERT_EQ(test_spec.solve_groups[2].matrix_type, "dense");
    ASSERT_EQ(test_spec.solve_groups[2].solver_args.max_iter, 10);
    ASSERT_EQ(test_spec.solve_groups[2].solver_args.max_inner_iter, 3);
    ASSERT_EQ(test_spec.solve_groups[2].solver_args.target_rel_res, 1e-10);
    ASSERT_EQ(
        test_spec.solve_groups[2].precond_specs,
        Preconditioner_Spec("none")
    );
    ASSERT_EQ(test_spec.solve_groups[2].matrices_to_test.size(), 2);
    ASSERT_EQ(test_spec.solve_groups[2].matrices_to_test[0], "662_bus.csv");
    ASSERT_EQ(test_spec.solve_groups[2].matrices_to_test[1], "685_bus.csv");

}

TEST_F(Test_Experiment_Read, TestBadJsonParse) {

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("doesnt_exist.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("not_a_json.txt")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_json_parse.json")
            );
        }
    );

}

TEST_F(Test_Experiment_Read, TestBadJsonSolveGroupMembers) {

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_invalid_solve_group.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_empty_solve_group.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_bad_key.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_missing_arg.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_repeated_key.json")
            );
        }
    );

}

TEST_F(Test_Experiment_Read, TestBadJsonSolveGroupMemberValues) {

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_int.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_double.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_solvers_to_use_type.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path(
                    "bad_solve_group_arg_bad_solvers_to_use_wrong_solverid"
                    ".json"
                )
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path(
                    "bad_solve_group_arg_bad_solvers_to_use_repeat_solverid"
                    ".json"
                )
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrix_type.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrix_type_wrong_str.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs_wrong_str.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs_shortarr.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs_longarr.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs_badname.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path(
                    "bad_solve_group_arg_bad_precond_specs_badilutptau.json"
                )
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_precond_specs_badilutpp.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrices_empty.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrices_type.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrices_member_type.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_bad_matrices_extension.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_solver_arg_pkg_badouter.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_solver_arg_pkg_badinner.json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_solver_arg_pkg_badrelres_type"
                         ".json")
            );
        }
    );

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        [this]() -> void {
            parse_experiment_spec(
                test_json_dir /
                fs::path("bad_solve_group_arg_solver_arg_pkg_badrelres_value"
                         ".json")
            );
        }
    );

}
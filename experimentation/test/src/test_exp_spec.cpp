#include "test_experiment.h"

#include "exp_spec/exp_spec.h"

class Test_Experiment_Spec: public Test_Experiment_Base
{
public:

    template <template <typename> typename TMatrix>
    void Test_Preconditioner_Spec_Basic_Construct() {

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

    template <template <typename> typename TMatrix>
    void Test_Preconditioner_Spec_Bad() {

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

    template <template <typename> typename TMatrix>
    void Test_Solve_Group_Basic_Construct() {

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

    template <template <typename> typename TMatrix>
    void Test_Solve_Group_Bad() {
        
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

};

TEST_F(Test_Experiment_Spec, Test_Preconditioner_Spec_Basic_Construct) {
    Test_Preconditioner_Spec_Basic_Construct<MatrixDense>();
    Test_Preconditioner_Spec_Basic_Construct<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Spec, Test_Preconditioner_Spec_Bad) {
    Test_Preconditioner_Spec_Bad<MatrixDense>();
    Test_Preconditioner_Spec_Bad<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Spec, Test_Solve_Group_Basic_Construct) {
    Test_Solve_Group_Basic_Construct<MatrixDense>();
    Test_Solve_Group_Basic_Construct<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Spec, Test_Solve_Group_Bad) {
    Test_Solve_Group_Bad<MatrixDense>();
    Test_Solve_Group_Bad<NoFillMatrixSparse>();
}
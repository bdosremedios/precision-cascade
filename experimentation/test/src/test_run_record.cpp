#include "test_experiment.h"

#include "experiment_run_record.h"

#include <map>

class TestRun: public TestExperimentBase
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
        } else if (solver_id == "SimpleConstantThreshold") {
            return "SimpleConstantThreshold";
        } else if (solver_id == "RestartCount") {
            return "RestartCount";
        } else {
            throw std::runtime_error("get_solver_str: no matching solver_id");
        }

    }

    std::string get_left_precond(
        std::string solver_id,
        Solve_Group_Precond_Specs precond_specs
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
        Solve_Group_Precond_Specs precond_specs
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

    TestRun() {
        logger = Experiment_Log();
    }

    ~TestRun() {}

    template <template <typename> typename TMatrix>
    void Test_Load_Lin_Sys(std::string matrix_name) {

        GenericLinearSystem<TMatrix> gen_lin_sys = load_lin_sys<TMatrix>(
            *TestExperimentBase::cu_handles_ptr,
            test_data_dir,
            matrix_name,
            logger
        );

        fs::path matrix_path(test_data_dir / fs::path(matrix_name));
        TMatrix<double> target_A(*TestExperimentBase::cu_handles_ptr);
        if (matrix_path.extension() == ".mtx") {
            target_A = read_matrixMTX<TMatrix, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        } else if (matrix_path.extension() == ".csv") {
            target_A = read_matrixCSV<TMatrix, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        } else {
            FAIL();
        }
        target_A.normalize_magnitude();

        ASSERT_EQ(gen_lin_sys.get_A().rows(), target_A.rows());
        ASSERT_EQ(gen_lin_sys.get_A().cols(), target_A.cols());
        for (int j=0; j<target_A.cols(); ++j) {
            for (int i=0; i<target_A.rows(); ++i) {
                ASSERT_EQ(
                    gen_lin_sys.get_A().get_elem(i, j),
                    target_A.get_elem(i, j)
                );
            }
        }
    
        ASSERT_EQ(gen_lin_sys.get_b().rows(), target_A.rows());

    }

    template <template <typename> typename TMatrix>
    void Test_Load_Lin_Sys_w_RHS(std::string matrix_name) {

        GenericLinearSystem<TMatrix> gen_lin_sys = load_lin_sys<TMatrix>(
            *TestExperimentBase::cu_handles_ptr,
            test_data_dir,
            matrix_name,
            logger
        );

        fs::path matrix_path(test_data_dir / fs::path(matrix_name));
        TMatrix<double> target_A(*TestExperimentBase::cu_handles_ptr);
        if (matrix_path.extension() == fs::path(".mtx")) {
            target_A = read_matrixMTX<TMatrix, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        } else if (matrix_path.extension() == fs::path(".csv")) {
            target_A = read_matrixCSV<TMatrix, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        } else {
            FAIL();
        }
        Scalar<double> A_max_mag = target_A.get_max_mag_elem();
        target_A.normalize_magnitude();

        ASSERT_EQ(gen_lin_sys.get_A().rows(), target_A.rows());
        ASSERT_EQ(gen_lin_sys.get_A().cols(), target_A.cols());
        for (int j=0; j<target_A.cols(); ++j) {
            for (int i=0; i<target_A.rows(); ++i) {
                ASSERT_EQ(
                    gen_lin_sys.get_A().get_elem(i, j),
                    target_A.get_elem(i, j)
                );
            }
        }
    
        ASSERT_EQ(gen_lin_sys.get_b().rows(), target_A.rows());

        fs::path matrix_path_b = (
            test_data_dir /
            fs::path(
                matrix_path.stem().string() + "_b" +
                matrix_path.extension().string()
            )
        );
        if (matrix_path_b.extension() == fs::path(".mtx")) {

            TMatrix<double> target_b(read_matrixMTX<TMatrix, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path_b
            ));
            target_b /= A_max_mag;
            bool matches_one_col = false;
            for (int j=0; j<target_b.cols(); ++j) {
                matches_one_col = (
                    matches_one_col ||
                    (gen_lin_sys.get_b() == target_b.get_col(j).copy_to_vec())
                );
            }
            ASSERT_TRUE(matches_one_col);

        } else if (matrix_path_b.extension() == fs::path(".csv")) {

            Vector<double> target_b(read_vectorCSV<double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path_b
            ));
            target_b /= A_max_mag;
            ASSERT_EQ(gen_lin_sys.get_b(), target_b);

        } else {
            FAIL();
        }

    }

    template <template <typename> typename TMatrix>
    void Test_Mismatch_Load_Lin_Sys_w_RHS(std::string matrix_name) {

        auto test_func = [matrix_name, this]() -> void {
            GenericLinearSystem<TMatrix> gen_lin_sys = (
                load_lin_sys<TMatrix>(
                    *TestExperimentBase::cu_handles_ptr,
                    test_data_dir,
                    matrix_name,
                    logger
                )
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, test_func);

    }

    template <template <typename> typename TMatrix>
    void Test_Run_Solve_Group(Solve_Group solve_group) {

        run_record_solve_group<TMatrix>(
            *TestExperimentBase::cu_handles_ptr,
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
                    // ASSERT_TRUE(
                    //     contains_strings(
                    //         loaded_file["precond_left"],
                    //         std::vector<std::string>(
                    //             {get_type_str(solver_id),
                    //              get_mat_type_str(solve_group.matrix_type),
                    //              get_left_precond(
                    //                 solver_id,
                    //                 solve_group.precond_specs
                    //              )}
                    //         )
                    //     )
                    // );
                    // ASSERT_TRUE(
                    //     contains_strings(
                    //         loaded_file["precond_right"],
                    //         std::vector<std::string>(
                    //             {get_type_str(solver_id),
                    //              get_mat_type_str(solve_group.matrix_type),
                    //              get_right_precond(
                    //                 solver_id,
                    //                 solve_group.precond_specs
                    //              )}
                    //         )
                    //     )
                    // );
                    // ASSERT_EQ(
                    //     loaded_file["precond_specs"],
                    //     solve_group.precond_specs.get_spec_string()
                    // );
            
                    ASSERT_EQ(loaded_file["initiated"], "true");
                    ASSERT_EQ(loaded_file["terminated"], "true");

                }
            }
        }

    }

    template <template <typename> typename TMatrix>
    void Test_Run_Experiment_Specification(Experiment_Specification exp_spec) {

        run_record_experimental_spec(
            *TestExperimentBase::cu_handles_ptr,
            exp_spec,
            test_data_dir,
            test_output_dir,
            logger
        );

    }

};

TEST_F(TestRun, Test_Load_Lin_Sys) {

    Test_Load_Lin_Sys<MatrixDense>("easy_4_4.csv");
    Test_Load_Lin_Sys<NoFillMatrixSparse>("easy_4_4.csv");

    Test_Load_Lin_Sys<MatrixDense>("easy_4_4.mtx");
    Test_Load_Lin_Sys<NoFillMatrixSparse>("easy_4_4.mtx");
    
}

TEST_F(TestRun, Test_Load_Lin_Sys_w_RHS) {

    Test_Load_Lin_Sys_w_RHS<MatrixDense>("paired_mat.csv");
    Test_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>("paired_mat.csv");

    Test_Load_Lin_Sys_w_RHS<MatrixDense>("paired_mat.mtx");
    Test_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>("paired_mat.mtx");
    
}

TEST_F(TestRun, Test_Mismatch_Load_Lin_Sys_w_RHS) {

    Test_Mismatch_Load_Lin_Sys_w_RHS<MatrixDense>(
        "bad_paired_mat_small.csv"
    );
    Test_Mismatch_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>(
        "bad_paired_mat_small.csv"
    );

    Test_Mismatch_Load_Lin_Sys_w_RHS<MatrixDense>(
        "bad_paired_mat_small.mtx"
    );
    Test_Mismatch_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>(
        "bad_paired_mat_small.mtx"
    );

    Test_Mismatch_Load_Lin_Sys_w_RHS<MatrixDense>(
        "bad_paired_mat_big.csv"
    );
    Test_Mismatch_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>(
        "bad_paired_mat_big.csv"
    );

    Test_Mismatch_Load_Lin_Sys_w_RHS<MatrixDense>(
        "bad_paired_mat_big.mtx"
    );
    Test_Mismatch_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>(
        "bad_paired_mat_big.mtx"
    );
    
}

TEST_F(TestRun, Test_AllSolvers_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "allsolvers_dense",
        std::vector<std::string>(
            {"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}
        ),
        "dense", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "allsolvers_sparse",
        std::vector<std::string>(
            {"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}
        ),
        "sparse", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

}

TEST_F(TestRun, Test_AllPreconditioners_Run_Solve_Group) {

    std::vector<Solve_Group_Precond_Specs> precond_spec_vec {
        Solve_Group_Precond_Specs("none"),
        Solve_Group_Precond_Specs("jacobi"),
        Solve_Group_Precond_Specs("ilu0"),
        Solve_Group_Precond_Specs("ilutp", 1e-4, 20)
    };

    for (Solve_Group_Precond_Specs precond_specs : precond_spec_vec) {

        Solve_Group solve_group_dense(
            "allpreconditioners_dense_"+precond_specs.name,
            std::vector<std::string>({"FP16", "FP32", "FP64", "RestartCount"}),
            "dense", 3, 10, 4, 1e-4,
            precond_specs,
            std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
        );

        Test_Run_Solve_Group<MatrixDense>(solve_group_dense);

        Solve_Group solve_group_sparse(
            "allpreconditioners_sparse_"+precond_specs.name,
            std::vector<std::string>({"FP16", "FP32", "FP64", "RestartCount"}),
            "sparse", 3, 10, 4, 1e-4,
            precond_specs,
            std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
        );

        Test_Run_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

    }

}

TEST_F(TestRun, Test_Mix_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "mixsolvers_dense",
        std::vector<std::string>({"FP64", "FP16", "SimpleConstantThreshold"}),
        "dense", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "mixsolvers_sparse",
        std::vector<std::string>({"FP64", "FP16", "SimpleConstantThreshold"}),
        "sparse", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Solve_Group<NoFillMatrixSparse>(solve_group_sparse);

}
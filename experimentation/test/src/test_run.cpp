#include <map>

#include "test_experiment.h"

#include "experiment_run.h"

class TestRun: public TestExperimentBase
{
private:

    std::string get_type_str(std::string solver_id) {
        if (solver_id == "FP16") {
            return "struct __half";
        } else if (solver_id == "FP32") {
            return "float";
        } else {
            return "double";
        }
    }

    std::string get_mat_type_str(std::string mat_type_id) {
        if (mat_type_id == "dense") {
            return "class MatrixDense";
        } else if (mat_type_id == "sparse") {
            return "class NoFillMatrixSparse";
        } else {
            throw std::runtime_error("get_mat_type_str: no matching mat_type_id");
        }
    }

    std::string get_solver_str(std::string mat_type_id, std::string solver_id) {

        std::string type_str = get_type_str(solver_id);
        std::string mat_type_str = get_mat_type_str(mat_type_id);

        if ((solver_id == "FP16") || (solver_id == "FP32") || (solver_id == "FP64")) {
            return std::format("class FP_GMRES_IR_Solve<{},{}>", mat_type_str, type_str);
        } else if (solver_id == "SimpleConstantThreshold") {
            return std::format("class SimpleConstantThreshold<{}>", mat_type_str);
        } else if (solver_id == "RestartCount") {
            return std::format("class RestartCount<{}>", mat_type_str);
        } else {
            throw std::runtime_error("get_solver_str: no matching solver_id");
        }

    }

    std::string get_left_precond(
        std::string mat_type_id, std::string solver_id, Solve_Group_Precond_Specs precond_specs
    ) {

        std::string type_str = get_type_str(solver_id);
        std::string mat_type_str = get_mat_type_str(mat_type_id);

        if (precond_specs.name == "none") {
            return std::format("class NoPreconditioner<{},{}>", mat_type_str, type_str);
        } else if (precond_specs.name == "jacobi") {
            if ((solver_id == "FP16") || (solver_id == "FP32")) {
                return std::format("class MatrixInversePreconditioner<{},{}>", mat_type_str, type_str);
            } else {
                return std::format("class JacobiPreconditioner<{},{}>", mat_type_str, type_str);
            }
        } else if ((precond_specs.name == "ilu0") || (precond_specs.name == "ilutp")) {
            return std::format("class ILUPreconditioner<{},{}>", mat_type_str, type_str);
        } else {
            throw std::runtime_error("get_left_precond: no matching precond_specs");
        }

    }

    std::string get_right_precond(
        std::string mat_type_id, std::string solver_id, Solve_Group_Precond_Specs precond_specs
    ) {

        std::string type_str = get_type_str(solver_id);
        std::string mat_type_str = get_mat_type_str(mat_type_id);
        
        return std::format("class NoPreconditioner<{},{}>", mat_type_str, type_str);

    }

public:

    Experiment_Log logger;

    TestRun() {
        logger = Experiment_Log();
    }

    ~TestRun() {}

    template <template <typename> typename M>
    void Test_Load_Linear_Problem(std::string matrix_name) {
        
        LinSysSolnPair<M> pair = load_linear_problem<M>(
            *TestExperimentBase::cu_handles_ptr,
            test_data_dir,
            matrix_name,
            logger
        );

        fs::path matrix_path(test_data_dir / fs::path(matrix_name));
        M<double> target_A(*TestExperimentBase::cu_handles_ptr);
        if (matrix_path.extension() == ".mtx") {
            target_A = read_matrixMTX<M, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        } else if (matrix_path.extension() == ".csv") {
            target_A = read_matrixCSV<M, double>(
                *TestExperimentBase::cu_handles_ptr, matrix_path
            );
        }
        target_A.normalize_magnitude();

        ASSERT_EQ(pair.first.get_A().rows(), target_A.rows());
        ASSERT_EQ(pair.first.get_A().cols(), target_A.cols());
        for (int j=0; j<target_A.cols(); ++j) {
            for (int i=0; i<target_A.rows(); ++i) {
                ASSERT_EQ(pair.first.get_A().get_elem(i, j), target_A.get_elem(i, j));
            }
        }

        Vector<double> target_b(pair.first.get_A()*pair.second);
        ASSERT_EQ(pair.first.get_b(), target_b);
        for (int i=0; i<target_b.rows(); ++i) {
            ASSERT_EQ(pair.first.get_b().get_elem(i), target_b.get_elem(i));
        }

    }

    template <template <typename> typename M>
    void Test_Run_Solve_Group(Solve_Group solve_group) {

        run_solve_group<M>(
            *TestExperimentBase::cu_handles_ptr, solve_group, test_data_dir, test_output_dir, logger
        );

        fs::path solve_group_dir(test_output_dir / fs::path(solve_group.id));

        // Test that the right thing was run in each case
        for (std::string matrix_str : solve_group.matrices_to_test) {
            for (std::string solver_id : solve_group.solvers_to_use) {
                for (int exp_iter=0; exp_iter<solve_group.experiment_iterations; ++exp_iter) {

                    std::string experiment_id(
                        std::format(
                            "{}_{}_{}_{}",
                            matrix_str,
                            solver_id,
                            solve_group.precond_specs.get_spec_string(),
                            exp_iter
                        )
                    );
                    fs::path file_path = solve_group_dir / fs::path(experiment_id+".json");
                    std::ifstream file_in(file_path);

                    if (!(file_in.is_open())) {
                        throw std::runtime_error("Failed to read "+file_path.string());
                    }

                    json loaded_file = json::parse(file_in);

                    ASSERT_EQ(loaded_file["id"], experiment_id);
                    ASSERT_EQ(
                        loaded_file["solver_class"],
                        get_solver_str(solve_group.matrix_type, solver_id)
                    );
                    ASSERT_EQ(
                        loaded_file["precond_left"],
                        get_left_precond(
                            solve_group.matrix_type, solver_id, solve_group.precond_specs
                        )
                    );
                    ASSERT_EQ(
                        loaded_file["precond_right"],
                        get_right_precond(
                            solve_group.matrix_type, solver_id, solve_group.precond_specs
                        )
                    );
                    ASSERT_EQ(
                        loaded_file["precond_specs"],
                        solve_group.precond_specs.get_spec_string()
                    );
            
                    ASSERT_EQ(loaded_file["initiated"], "true");
                    ASSERT_EQ(loaded_file["terminated"], "true");

                }
            }
        }

    }

    template <template <typename> typename M>
    void Test_Run_Experiment_Specification(Experiment_Specification exp_spec) {

        run_experimental_spec(
            *TestExperimentBase::cu_handles_ptr, exp_spec, test_data_dir, test_output_dir, logger
        );

    }

};

TEST_F(TestRun, TestLoadLinearProblem) {

    Test_Load_Linear_Problem<MatrixDense>("easy_4_4.csv");
    Test_Load_Linear_Problem<NoFillMatrixSparse>("easy_4_4.csv");

    Test_Load_Linear_Problem<MatrixDense>("easy_4_4.mtx");
    Test_Load_Linear_Problem<NoFillMatrixSparse>("easy_4_4.mtx");
    
}

TEST_F(TestRun, Test_AllSolvers_Run_Solve_Group) {

    Solve_Group solve_group_dense(
        "allsolvers_dense",
        std::vector<std::string>({"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}),
        "dense", 3, 10, 4, 1e-4,
        Solve_Group_Precond_Specs("none"),
        std::vector<std::string>({"easy_4_4.csv", "easy_5_5.csv"})
    );

    Test_Run_Solve_Group<MatrixDense>(solve_group_dense);

    Solve_Group solve_group_sparse(
        "allsolvers_sparse",
        std::vector<std::string>({"FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"}),
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
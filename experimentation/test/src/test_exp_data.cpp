#include "test_experiment.h"

#include "exp_spec/exp_spec.h"
#include "exp_data/exp_data.h"
#include "exp_run/exp_run_record.h"

#include "tools/TypeIdentity.h"

#include <nlohmann/json.hpp>

#include <cuda_fp16.h>

using json = nlohmann::json;

class Test_Experiment_Data: public Test_Experiment_Base
{
private:

    MatrixDense<double> A = MatrixDense<double>(cuHandleBundle());
    Vector<double> b = Vector<double>(cuHandleBundle());
    const double u_dbl = std::pow(2, -52);
    SolveArgPkg solve_args;
    Experiment_Log logger;

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

    std::string get_tag(TMatrixIdentity<MatrixDense> _) {
        return "dense";
    }
    std::string get_tag(TMatrixIdentity<NoFillMatrixSparse> _) {
        return "sparse";
    }

    std::string get_type_str(TypeIdentity<__half> _) {
        return "__half";
    }
    std::string get_type_str(TypeIdentity<float> _) {
        #ifdef WIN32
        return "float";
        #else
        return "f";
        #endif
    }
    std::string get_type_str(TypeIdentity<double> _) {
        #ifdef WIN32
        return "double";
        #else
        return "d";
        #endif
    }

    std::string get_mat_type_str(TMatrixIdentity<MatrixDense> _) {
        return "MatrixDense";
    }
    std::string get_mat_type_str(TMatrixIdentity<NoFillMatrixSparse> _) {
        return "NoFillMatrixSparse";
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

    std::string get_left_precond(Preconditioner_Spec precond_specs) {

        if (precond_specs.name == "none") {
            return "NoPreconditioner";
        } else if (precond_specs.name == "jacobi") {
            return "JacobiPreconditioner";
        } else if (
            (precond_specs.name == "ilu0") || (precond_specs.name == "ilutp")
        ) {
            return "ILUPreconditioner";
        } else {
            throw std::runtime_error(
                "get_left_precond: no matching precond_specs"
            );
        }

    }

    std::string get_right_precond(Preconditioner_Spec precond_specs) {
        return "NoPreconditioner";

    }

    std::string bool_to_string(bool b) {
        return (b) ? "true" : "false";
    }

    template <template <typename> typename TMatrix>
    void ASSERT_MATCH_PRECOND_DATA(
        fs::path json_file, Preconditioner_Data<TMatrix> precond_data
    ) {

        std::ifstream file_in(json_file);
        if (!file_in.is_open()) { FAIL(); }

        json loaded_file = json::parse(file_in);

        ASSERT_EQ(loaded_file["id"], precond_data.id);
        ASSERT_TRUE(
            contains_strings(
                loaded_file["precond_left"],
                std::vector<std::string>(
                    {get_type_str(TypeIdentity<double>()),
                     get_mat_type_str(TMatrixIdentity<TMatrix>()),
                     get_left_precond(precond_data.precond_specs)}
                )
            )
        );
        ASSERT_TRUE(
            contains_strings(
                loaded_file["precond_right"],
                std::vector<std::string>(
                    {get_type_str(TypeIdentity<double>()),
                     get_mat_type_str(TMatrixIdentity<TMatrix>()),
                     get_right_precond(precond_data.precond_specs)}
                )
            )
        );
        ASSERT_EQ(
            loaded_file["precond_specs"],
            precond_data.precond_specs.get_spec_string()
        );
        ASSERT_EQ(
            loaded_file["elapsed_time_ms"],
            precond_data.clock.get_elapsed_time_ms()
        );

        file_in.close();

    }

public:

    Test_Experiment_Data() {
        A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
        b = A*Vector<double>::Random(*cu_handles_ptr, 16);
        logger = Experiment_Log();
    }

    ~Test_Experiment_Data() {}

    template <template <typename> typename TMatrix>
    void Test_Load_Lin_Sys(std::string matrix_name) {

        GenericLinearSystem<TMatrix> gen_lin_sys = load_lin_sys<TMatrix>(
            *Test_Experiment_Base::cu_handles_ptr,
            test_data_dir,
            matrix_name,
            logger
        );

        fs::path matrix_path(test_data_dir / fs::path(matrix_name));
        TMatrix<double> target_A(*Test_Experiment_Base::cu_handles_ptr);
        if (matrix_path.extension() == ".mtx") {
            target_A = read_matrixMTX<TMatrix, double>(
                *Test_Experiment_Base::cu_handles_ptr, matrix_path
            );
        } else if (matrix_path.extension() == ".csv") {
            target_A = read_matrixCSV<TMatrix, double>(
                *Test_Experiment_Base::cu_handles_ptr, matrix_path
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
            *Test_Experiment_Base::cu_handles_ptr,
            test_data_dir,
            matrix_name,
            logger
        );

        fs::path matrix_path(test_data_dir / fs::path(matrix_name));
        TMatrix<double> target_A(*Test_Experiment_Base::cu_handles_ptr);
        if (matrix_path.extension() == fs::path(".mtx")) {
            target_A = read_matrixMTX<TMatrix, double>(
                *Test_Experiment_Base::cu_handles_ptr, matrix_path
            );
        } else if (matrix_path.extension() == fs::path(".csv")) {
            target_A = read_matrixCSV<TMatrix, double>(
                *Test_Experiment_Base::cu_handles_ptr, matrix_path
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
                *Test_Experiment_Base::cu_handles_ptr, matrix_path_b
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
                *Test_Experiment_Base::cu_handles_ptr, matrix_path_b
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
                    *Test_Experiment_Base::cu_handles_ptr,
                    test_data_dir,
                    matrix_name,
                    logger
                )
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, test_func);

    }

    template <template <typename> typename TMatrix>
    void TestRecordOutputJsonPrecond() {

        std::string tag = get_tag(TMatrixIdentity<TMatrix>());
        
        Preconditioner_Spec none_precond_specs("none");

        Experiment_Clock none_clock;
        none_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> none_precond_arg_pkg;
        none_clock.stop_clock_experiment();

        Preconditioner_Data<TMatrix> none_data(
            "none_" + tag + "_id",
            none_clock,
            none_precond_specs,
            none_precond_arg_pkg
        );
        std::string none_file_name = "none_" + tag + "_file";
        none_data.record_json(none_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(none_file_name + ".json"),
            none_data
        );
        
        Preconditioner_Spec jacobi_precond_specs("jacobi");

        Experiment_Clock jacobi_clock;
        jacobi_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> jacobi_precond_arg_pkg(
            std::make_shared<JacobiPreconditioner<TMatrix, double>>(
                TMatrix<double>(A)
            )
        );
        jacobi_clock.stop_clock_experiment();

        Preconditioner_Data<TMatrix> jacobi_data(
            "jacobi_" + tag + "_id",
            jacobi_clock,
            jacobi_precond_specs,
            jacobi_precond_arg_pkg
        );
        std::string jacobi_file_name = "jacobi_" + tag + "_file";
        jacobi_data.record_json(jacobi_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(jacobi_file_name + ".json"),
            jacobi_data
        );
        
        Preconditioner_Spec ilu0_precond_specs("ilu0");

        Experiment_Clock ilu0_clock;
        ilu0_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> ilu0_precond_arg_pkg(
            std::make_shared<ILUPreconditioner<TMatrix, double>>(
                TMatrix<double>(A)
            )
        );
        ilu0_clock.stop_clock_experiment();

        Preconditioner_Data<TMatrix> ilu0_data(
            "ilu0_" + tag + "_id",
            ilu0_clock,
            ilu0_precond_specs,
            ilu0_precond_arg_pkg
        );
        std::string ilu0_file_name = "ilu0_" + tag + "_file";
        ilu0_data.record_json(ilu0_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(ilu0_file_name + ".json"),
            ilu0_data
        );
        
        Preconditioner_Spec ilutp_precond_specs(
            "ilutp", 1e-4, 50
        );

        Experiment_Clock ilutp_clock;
        ilutp_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> ilutp_precond_arg_pkg(
            std::make_shared<ILUPreconditioner<TMatrix, double>>(
                TMatrix<double>(A),
                ilutp_precond_specs.ilutp_tau,
                ilutp_precond_specs.ilutp_p,
                true
            )
        );
        ilutp_clock.stop_clock_experiment();

        Preconditioner_Data<TMatrix> ilutp_data(
            "ilutp_" + tag + "_id",
            ilutp_clock,
            ilutp_precond_specs,
            ilutp_precond_arg_pkg
        );
        std::string ilutp_file_name = "ilutp_" + tag + "_file";
        ilutp_data.record_json(ilutp_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(ilutp_file_name + ".json"),
            ilutp_data
        );

    }

    template <template <typename> typename TMatrix>
    void TestRecordOutputJsonFPGMRES() {

        std::string tag = get_tag(TMatrixIdentity<TMatrix>());
        std::string id = "fpgmres_" + tag + "_id";
        std::string file_name = "fpgmres_" + tag + "_file";

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, double> typ_lin_sys(&gen_lin_sys);

        std::shared_ptr<FP_GMRES_IR_Solve<TMatrix, double>> solve_ptr;
        solve_ptr = std::make_shared<FP_GMRES_IR_Solve<TMatrix, double>>(
            &typ_lin_sys, u_dbl, solve_args
        );

        Solve_Data<GenericIterativeSolve, TMatrix> data(
            execute_solve<GenericIterativeSolve, TMatrix>(
                id, solve_ptr, logger, false
            )
        );
        data.record_json(file_name, test_output_dir, logger);

        fs::path file_path = test_output_dir / fs::path(file_name + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], id);
        ASSERT_EQ(loaded_file["solver_class"], typeid(*solve_ptr).name());
        ASSERT_EQ(
            loaded_file["initiated"],
            bool_to_string(solve_ptr->check_initiated())
        );
        ASSERT_EQ(
            loaded_file["converged"],
            bool_to_string(solve_ptr->check_converged())
        );
        ASSERT_EQ(
            loaded_file["terminated"],
            bool_to_string(solve_ptr->check_terminated())
        );
        ASSERT_EQ(loaded_file["iteration"], solve_ptr->get_iteration());
        ASSERT_EQ(
            loaded_file["elapsed_time_ms"],
            data.clock.get_elapsed_time_ms()
        );

        std::vector<double> res_norm_history = (
            solve_ptr->get_res_norm_history()
        );
        for (int i=0; i<res_norm_history.size(); ++i) {
            ASSERT_EQ(loaded_file["res_norm_history"][i], res_norm_history[i]);
        }

        file_in.close();

    }

    template <template <typename> typename TMatrix>
    void TestRecordOutputJsonMPGMRES() {

        std::string tag = get_tag(TMatrixIdentity<TMatrix>());
        std::string id = "mpgmres_" + tag + "_id";
        std::string file_name = "mpgmres_" + tag + "_file";

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);

        std::shared_ptr<MP_GMRES_IR_Solve<TMatrix>> solve_ptr;
        solve_ptr = std::make_shared<SimpleConstantThreshold<TMatrix>>(
            &gen_lin_sys, solve_args
        );

        Solve_Data<MP_GMRES_IR_Solve, TMatrix> data(
            execute_solve<MP_GMRES_IR_Solve, TMatrix>(
                id, solve_ptr, logger, false
            )
        );
        data.record_json(file_name, test_output_dir, logger);

        fs::path file_path = test_output_dir / fs::path(file_name + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], id);
        ASSERT_EQ(loaded_file["solver_class"], typeid(*solve_ptr).name());
        ASSERT_EQ(
            loaded_file["initiated"],
            bool_to_string(solve_ptr->check_initiated())
        );
        ASSERT_EQ(
            loaded_file["converged"],
            bool_to_string(solve_ptr->check_converged())
        );
        ASSERT_EQ(
            loaded_file["terminated"],
            bool_to_string(solve_ptr->check_terminated())
        );
        ASSERT_EQ(loaded_file["iteration"], solve_ptr->get_iteration());
        ASSERT_EQ(
            loaded_file["elapsed_time_ms"],
            data.clock.get_elapsed_time_ms()
        );

        std::vector<double> res_norm_history = (
            solve_ptr->get_res_norm_history()
        );
        for (int i=0; i<res_norm_history.size(); ++i) {
            ASSERT_EQ(loaded_file["res_norm_history"][i], res_norm_history[i]);
        }

        file_in.close();

    }

};

TEST_F(Test_Experiment_Data, Test_Load_Lin_Sys) {

    Test_Load_Lin_Sys<MatrixDense>("easy_4_4.csv");
    Test_Load_Lin_Sys<NoFillMatrixSparse>("easy_4_4.csv");

    Test_Load_Lin_Sys<MatrixDense>("easy_4_4.mtx");
    Test_Load_Lin_Sys<NoFillMatrixSparse>("easy_4_4.mtx");
    
}

TEST_F(Test_Experiment_Data, Test_Load_Lin_Sys_w_RHS) {

    Test_Load_Lin_Sys_w_RHS<MatrixDense>("paired_mat.csv");
    Test_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>("paired_mat.csv");

    Test_Load_Lin_Sys_w_RHS<MatrixDense>("paired_mat.mtx");
    Test_Load_Lin_Sys_w_RHS<NoFillMatrixSparse>("paired_mat.mtx");
    
}

TEST_F(Test_Experiment_Data, Test_Mismatch_Load_Lin_Sys_w_RHS) {

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

TEST_F(Test_Experiment_Data, TestRecordOutputJsonPrecond) {
    TestRecordOutputJsonPrecond<MatrixDense>();
    TestRecordOutputJsonPrecond<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Data, TestRecordOutputJsonFPGMRES) {
    TestRecordOutputJsonFPGMRES<MatrixDense>();
    TestRecordOutputJsonFPGMRES<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Data, TestRecordOutputJsonMPGMRES) {
    TestRecordOutputJsonMPGMRES<MatrixDense>();
    TestRecordOutputJsonMPGMRES<NoFillMatrixSparse>();
}
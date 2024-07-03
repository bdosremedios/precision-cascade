#include "test_experiment.h"

#include "exp_spec/exp_spec.h"
#include "exp_data/exp_data.h"
#include "exp_run/exp_run_record.h"

#include "tools/TypeIdentity.h"

#include <nlohmann/json.hpp>

#include <cuda_fp16.h>

using json = nlohmann::json;

class Test_Experiment_Record: public TestExperimentBase
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

    Test_Experiment_Record() {
        A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
        b = A*Vector<double>::Random(*cu_handles_ptr, 16);
        logger = Experiment_Log();
    }

    ~Test_Experiment_Record() {}

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
            execute_solve<GenericIterativeSolve, TMatrix>(id, solve_ptr, false)
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
            execute_solve<MP_GMRES_IR_Solve, TMatrix>(id, solve_ptr, false)
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

TEST_F(Test_Experiment_Record, TestRecordOutputJsonPrecond) {
    TestRecordOutputJsonPrecond<MatrixDense>();
    TestRecordOutputJsonPrecond<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Record, TestRecordOutputJsonFPGMRES) {
    TestRecordOutputJsonFPGMRES<MatrixDense>();
    TestRecordOutputJsonFPGMRES<NoFillMatrixSparse>();
}

TEST_F(Test_Experiment_Record, TestRecordOutputJsonMPGMRES) {
    TestRecordOutputJsonMPGMRES<MatrixDense>();
    TestRecordOutputJsonMPGMRES<NoFillMatrixSparse>();
}
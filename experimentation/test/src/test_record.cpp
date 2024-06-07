#include "test_experiment.h"

#include "experiment_run.h"
#include "experiment_record.h"

#include "tools/cuHandleBundle.h"
#include "types/types.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

class TestRecord: public TestExperimentBase
{
public:

    SolveArgPkg solve_args;
    MatrixDense<double> A = MatrixDense<double>(cuHandleBundle());
    Vector<double> b = Vector<double>(cuHandleBundle());
    const double u_dbl = std::pow(2, -52);
    Experiment_Log logger;

    TestRecord() {
        A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
        b = A*Vector<double>::Random(*cu_handles_ptr, 16);
        logger = Experiment_Log();
    }

    ~TestRecord() {}

    std::string bool_to_string(bool b) {
        if (b) {
            return "true";
        } else {
            return "false";
        }
    }

    template <template <typename> typename M>
    void TestRecordOutputJsonFPGMRES(std::string file_name) {

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, double> typ_lin_sys(&gen_lin_sys);

        Solve_Group_Precond_Specs sg_precond_specs("none");
        PrecondArgPkg<M, double> precond_arg_pkg;

        std::shared_ptr<FP_GMRES_IR_Solve<M, double>> solve_ptr;
        solve_ptr = std::make_shared<FP_GMRES_IR_Solve<M, double>>(
            &typ_lin_sys, u_dbl, solve_args, precond_arg_pkg
        );

        Experiment_Data<GenericIterativeSolve, M> data(
            execute_solve<GenericIterativeSolve, M>(solve_ptr, false)
        );

        record_FPGMRES_data_json(
            data, precond_arg_pkg, sg_precond_specs.get_spec_string(),
            file_name, test_output_dir, logger
        );

        fs::path file_path = test_output_dir / fs::path(file_name + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], file_name);
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

        ASSERT_EQ(
            loaded_file["precond_left"],
            typeid(*precond_arg_pkg.left_precond).name()
        );
        ASSERT_EQ(
            loaded_file["precond_right"],
            typeid(*precond_arg_pkg.right_precond).name()
        );
        ASSERT_EQ(
            loaded_file["precond_specs"],
            sg_precond_specs.get_spec_string()
        );

        std::vector<double> res_norm_history = (
            solve_ptr->get_res_norm_history()
        );
        for (int i=0; i<res_norm_history.size(); ++i) {
            ASSERT_EQ(loaded_file["res_norm_history"][i], res_norm_history[i]);
        }

        file_in.close();

    }

    template <template <typename> typename M>
    void TestRecordOutputJsonMPGMRES(std::string file_name) {

        GenericLinearSystem<M> gen_lin_sys(A, b);

        Solve_Group_Precond_Specs sg_precond_specs("none");
        PrecondArgPkg<M, double> precond_arg_pkg;

        std::shared_ptr<MP_GMRES_IR_Solve<M>> solve_ptr;
        solve_ptr = std::make_shared<SimpleConstantThreshold<M>>(
            &gen_lin_sys, solve_args
        );

        Experiment_Data<MP_GMRES_IR_Solve, M> data(
            execute_solve<MP_GMRES_IR_Solve, M>(solve_ptr, false)
        );

        record_MPGMRES_data_json(
            data, precond_arg_pkg, sg_precond_specs.get_spec_string(),
            file_name, test_output_dir, logger
        );

        fs::path file_path = test_output_dir / fs::path(file_name + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], file_name);
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

        ASSERT_EQ(
            loaded_file["precond_left"],
            typeid(*precond_arg_pkg.left_precond).name()
        );
        ASSERT_EQ(
            loaded_file["precond_right"],
            typeid(*precond_arg_pkg.right_precond).name()
        );
        ASSERT_EQ(
            loaded_file["precond_specs"],
            sg_precond_specs.get_spec_string()
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

TEST_F(TestRecord, TestRecordOutputJsonFPGMRES) {
    TestRecordOutputJsonFPGMRES<MatrixDense>("FPGMRESTestRecord_Dense");
    TestRecordOutputJsonFPGMRES<NoFillMatrixSparse>("FPGMRESTestRecord_Sparse");
}

TEST_F(TestRecord, TestRecordOutputJsonMPGMRES) {
    TestRecordOutputJsonMPGMRES<MatrixDense>("MPGMRESTestRecord_Dense");
    TestRecordOutputJsonMPGMRES<NoFillMatrixSparse>("MPGMRESTestRecord_Sparse");
}
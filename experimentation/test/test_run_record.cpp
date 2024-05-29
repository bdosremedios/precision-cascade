#include "test_experiment.h"

#include <filesystem>
#include <fstream>
#include <cstdio>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <nlohmann/json.hpp>

#include "tools/cuHandleBundle.h"
#include "types/types.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

#include "experiment_run.h"
#include "experiment_record.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class TestRecord: public TestExperimentBase
{
public:

    cuHandleBundle *cu_handles_ptr;
    SolveArgPkg solve_args;
    MatrixDense<double> A = MatrixDense<double>(cuHandleBundle());
    Vector<double> b = Vector<double>(cuHandleBundle());
    const double u_dbl = std::pow(2, -52);
    Experiment_Log logger;

    TestRecord() {
        cu_handles_ptr = new cuHandleBundle();
        cu_handles_ptr->create();
        A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
        b = A*Vector<double>::Random(*cu_handles_ptr, 16);
        logger = Experiment_Log();
    }

    ~TestRecord() {
        cu_handles_ptr->destroy();
        delete cu_handles_ptr;
    }

    std::string bool_to_string(bool b) {
        if (b) {
            return "true";
        } else {
            return "false";
        }
    }

    template <template <typename> typename M>
    void TestRunAndOutputJsonFPGMRES(std::string id) {

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, double> typ_lin_sys(&gen_lin_sys);

        std::shared_ptr<FP_GMRES_IR_Solve<M, double>> solve_ptr;
        solve_ptr = std::make_shared<FP_GMRES_IR_Solve<M, double>>(&typ_lin_sys, u_dbl, solve_args);

        Experiment_Data<GenericIterativeSolve, M> data(
            execute_solve<GenericIterativeSolve, M>(solve_ptr, false)
        );

        record_FPGMRES_experimental_data_json(data, id, test_json_dir, logger);

        fs::path file_path = test_json_dir / fs::path(id + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], id);
        ASSERT_EQ(loaded_file["solver_class"], typeid(*solve_ptr).name());
        ASSERT_EQ(loaded_file["initiated"], bool_to_string(solve_ptr->check_initiated()));
        ASSERT_EQ(loaded_file["converged"], bool_to_string(solve_ptr->check_converged()));
        ASSERT_EQ(loaded_file["terminated"], bool_to_string(solve_ptr->check_terminated()));
        ASSERT_EQ(loaded_file["iteration"], solve_ptr->get_iteration());
        ASSERT_EQ(loaded_file["elapsed_time_ms"], data.clock.get_elapsed_time_ms());

        std::vector<double> res_norm_hist = solve_ptr->get_res_norm_hist();
        for (int i=0; i<res_norm_hist.size(); ++i) {
            ASSERT_EQ(loaded_file["res_norm_hist"][i], res_norm_hist[i]);
        }

        MatrixDense<double> res_hist = solve_ptr->get_res_hist();

        file_in.close();

    }

    template <template <typename> typename M>
    void TestRunAndOutputJsonMPGMRES(std::string id) {

        GenericLinearSystem<M> gen_lin_sys(A, b);

        std::shared_ptr<MP_GMRES_IR_Solve<M>> solve_ptr;
        solve_ptr = std::make_shared<SimpleConstantThreshold<M>>(&gen_lin_sys, solve_args);

        Experiment_Data<MP_GMRES_IR_Solve, M> data(
            execute_solve<MP_GMRES_IR_Solve, M>(solve_ptr, false)
        );

        std::string id = "MPGMRESTestRecord";
        record_MPGMRES_experimental_data_json(data, id, test_json_dir, logger);

        fs::path file_path = test_json_dir / fs::path(id + ".json");
        std::ifstream file_in(file_path);

        json loaded_file = json::parse(file_in);
        ASSERT_EQ(loaded_file["id"], id);
        ASSERT_EQ(loaded_file["solver_class"], typeid(*solve_ptr).name());
        ASSERT_EQ(loaded_file["initiated"], bool_to_string(solve_ptr->check_initiated()));
        ASSERT_EQ(loaded_file["converged"], bool_to_string(solve_ptr->check_converged()));
        ASSERT_EQ(loaded_file["terminated"], bool_to_string(solve_ptr->check_terminated()));
        ASSERT_EQ(loaded_file["iteration"], solve_ptr->get_iteration());
        ASSERT_EQ(loaded_file["elapsed_time_ms"], data.clock.get_elapsed_time_ms());

        std::vector<double> res_norm_hist = solve_ptr->get_res_norm_hist();
        for (int i=0; i<res_norm_hist.size(); ++i) {
            ASSERT_EQ(loaded_file["res_norm_hist"][i], res_norm_hist[i]);
        }

        MatrixDense<double> res_hist = solve_ptr->get_res_hist();

        file_in.close();

    }

};

TEST_F(TestRecord, TestRunAndOutputJsonFPGMRES) {
    TestRunAndOutputJsonFPGMRES<MatrixDense>("FPGMRESTestRecord_Dense");
    TestRunAndOutputJsonFPGMRES<NoFillMatrixSparse>("FPGMRESTestRecord_Sparse");
}

TEST_F(TestRecord, TestRunAndOutputJsonMPGMRES) {
    TestRunAndOutputJsonFPGMRES<MatrixDense>("MPGMRESTestRecord_Dense");
    TestRunAndOutputJsonFPGMRES<NoFillMatrixSparse>("MPGMRESTestRecord_Sparse");
}
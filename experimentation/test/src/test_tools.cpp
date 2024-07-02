#include "test_experiment.h"

#include "experiment_recorders.h"
#include "experiment_run_record.h"
#include "experiment_log.h"

#include "tools/TypeIdentity.h"

#include <cuda_fp16.h>

class TestTools: public TestExperimentBase
{
private:

    MatrixDense<double> A = MatrixDense<double>(cuHandleBundle());
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

    std::string get_left_precond(Solve_Group_Precond_Specs precond_specs) {

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

    std::string get_right_precond(Solve_Group_Precond_Specs precond_specs) {
        return "NoPreconditioner";

    }

    template <template <typename> typename TMatrix>
    void ASSERT_MATCH_PRECOND_DATA(
        fs::path json_file, Precond_Data<TMatrix> precond_data
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

        file_in.close();

    }

public:

    TestTools() {
        A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
        logger = Experiment_Log();
    }

    ~TestTools() {}

    template <template <typename> typename TMatrix>
    void TestRecordOutputJsonPrecond(std::string tag) {
        
        Solve_Group_Precond_Specs none_precond_specs("none");

        Experiment_Clock none_clock;
        none_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> none_precond_arg_pkg;
        none_clock.stop_clock_experiment();

        Precond_Data<TMatrix> none_data(
            "none_id_" + tag,
            none_clock,
            none_precond_specs,
            none_precond_arg_pkg
        );
        std::string none_file_name = "none_file_" + tag;
        none_data.record_json(none_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(none_file_name + ".json"),
            none_data
        );
        
        Solve_Group_Precond_Specs jacobi_precond_specs("jacobi");

        Experiment_Clock jacobi_clock;
        jacobi_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> jacobi_precond_arg_pkg(
            std::make_shared<JacobiPreconditioner<TMatrix, double>>(
                TMatrix<double>(A)
            )
        );
        jacobi_clock.stop_clock_experiment();

        Precond_Data<TMatrix> jacobi_data(
            "jacobi_id_" + tag,
            jacobi_clock,
            jacobi_precond_specs,
            jacobi_precond_arg_pkg
        );
        std::string jacobi_file_name = "jacobi_file_" + tag;
        jacobi_data.record_json(jacobi_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(jacobi_file_name + ".json"),
            jacobi_data
        );
        
        Solve_Group_Precond_Specs ilu0_precond_specs("ilu0");

        Experiment_Clock ilu0_clock;
        ilu0_clock.start_clock_experiment();
        PrecondArgPkg<TMatrix, double> ilu0_precond_arg_pkg(
            std::make_shared<ILUPreconditioner<TMatrix, double>>(
                TMatrix<double>(A)
            )
        );
        ilu0_clock.stop_clock_experiment();

        Precond_Data<TMatrix> ilu0_data(
            "ilu0_id_" + tag,
            ilu0_clock,
            ilu0_precond_specs,
            ilu0_precond_arg_pkg
        );
        std::string ilu0_file_name = "ilu0_file_" + tag;
        ilu0_data.record_json(ilu0_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(ilu0_file_name + ".json"),
            ilu0_data
        );
        
        Solve_Group_Precond_Specs ilutp_precond_specs(
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

        Precond_Data<TMatrix> ilutp_data(
            "ilutp_id_" + tag,
            ilutp_clock,
            ilutp_precond_specs,
            ilutp_precond_arg_pkg
        );
        std::string ilutp_file_name = "ilutp_file_" + tag;
        ilutp_data.record_json(ilutp_file_name, test_output_dir, logger);

        ASSERT_MATCH_PRECOND_DATA(
            test_output_dir / fs::path(ilutp_file_name + ".json"),
            ilutp_data
        );

    }

};

TEST_F(TestTools, TestRecordOutputJsonPrecond) {
    TestRecordOutputJsonPrecond<MatrixDense>("dense");
    TestRecordOutputJsonPrecond<NoFillMatrixSparse>("sparse");
}
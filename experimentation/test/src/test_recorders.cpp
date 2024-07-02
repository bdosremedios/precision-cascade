// #include "test_experiment.h"

// #include "experiment_run_record.h"

// #include "tools/cuHandleBundle.h"
// #include "types/types.h"
// #include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"
// #include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

// #include <nlohmann/json.hpp>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// #include <filesystem>
// #include <fstream>
// #include <cstdio>
// #include <cmath>

// namespace fs = std::filesystem;
// using json = nlohmann::json;

// class TestRecord: public TestExperimentBase
// {
// private:

//     SolveArgPkg solve_args;
//     MatrixDense<double> A = MatrixDense<double>(cuHandleBundle());
//     Vector<double> b = Vector<double>(cuHandleBundle());
//     const double u_dbl = std::pow(2, -52);
//     Experiment_Log logger;

//     std::string bool_to_string(bool b) {
//         return (b) ? "true" : "false";
//     }

// public:

//     TestRecord() {
//         A = MatrixDense<double>::Random(*cu_handles_ptr, 16, 16);
//         b = A*Vector<double>::Random(*cu_handles_ptr, 16);
//         logger = Experiment_Log();
//     }

//     ~TestRecord() {}

// };
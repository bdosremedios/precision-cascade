#include <filesystem>
#include <iostream>
#include <cmath>
#include <string>

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include "types/types.h"
#include "tools/tools.h"

namespace fs = std::filesystem;
using std::cout, std::endl;
using std::pow;
using std::string;

const double u_hlf = pow(2, -10);
const double u_sgl = pow(2, -23);
const double u_dbl = pow(2, -52);

string get_file_name(fs::path file_path) {

    string temp = file_path.string();
    temp = temp.substr(temp.find_last_of("//")+1);
    temp = temp.substr(0, temp.find_last_of("."));

    return temp;

}

int main() {

    cout << "*** STARTING NUMERICAL EXPERIMENTATION ***" << endl;

    fs::path load_dir("/home/bdosre/dev/numerical_experimentation/data/experiment_matrices");

    fs::directory_iterator iter(load_dir);
    fs::directory_iterator curr = fs::begin(iter);

    // for (fs::directory_iterator curr = fs::begin(iter); curr != fs::end(iter); ++curr) {

        cout << "Testing: " << *curr << endl;

        MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(*curr);
        for (int i=1; i<=5; ++i) {

            string ID_prefix = get_file_name(*curr) + "_" + to_string(i);
            MatrixVector<double> b = A*MatrixVector<double>::Random(A.cols());

            SolveArgPkg args;
            args.init_guess = MatrixVector<double>::Zero(A.cols());
            args.max_iter = 40;
            args.max_inner_iter = 20;
            args.target_rel_res = pow(10, -10);

            FP_GMRES_IR_Solve<MatrixSparse, half> fpgmres_hlf(A, b, u_hlf, args);
            FP_GMRES_IR_Solve<MatrixSparse, float> fpgmres_sgl(A, b, u_sgl, args);
            FP_GMRES_IR_Solve<MatrixSparse, double> fpgmres_dbl(A, b, u_dbl, args);
            SimpleConstantThreshold<MatrixSparse> mpgmres(A, b, args);

        }

    // }

    cout << "*** FINISH NUMERICAL EXPERIMENTATION ***" << endl;
    
    return 0;

}
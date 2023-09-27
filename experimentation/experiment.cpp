#include <filesystem>
#include <iostream>
#include <string>

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

#include "types/types.h"
#include "tools/tools.h"

namespace fs = std::filesystem;
using std::cout, std::endl;
using std::string;

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

    for (fs::directory_iterator curr = fs::begin(iter); curr != fs::end(iter); ++curr) {

        cout << "Testing: " << *curr << endl;
        // MatrixSparse<double> A = read_matrixCSV<MatrixSparse, double>(*curr);
        for (int i=1; i<=5; ++i) {
            string ID = get_file_name(*curr) + "_" + to_string(i);
        }

    }

    cout << "*** FINISH NUMERICAL EXPERIMENTATION ***" << endl;
    
    return 0;

}
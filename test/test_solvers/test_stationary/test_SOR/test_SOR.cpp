#include "../../../test.h"

#include "solvers/stationary/SOR.h"

class SORTest: public TestBase
{
public:

    vector<double> ws{1.25, 1.5, 1.75};

    SolveArgPkg success_args;
    SolveArgPkg fail_args;

    void SetUp() {

        success_args = SolveArgPkg();
        success_args.max_iter = 1000;

        fail_args = SolveArgPkg();
        fail_args.max_iter = 300;

    }

};

TEST_F(SORTest, SolveConvDiff64_Double) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_dbl;
        SORSolve<double> SOR_solve_d(A, b, *w, success_args);
        SOR_solve_d.solve();
        if (*show_plots) { SOR_solve_d.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_d.check_converged());
        EXPECT_LE(SOR_solve_d.get_relres(), conv_tol_dbl);

    }

}

TEST_F(SORTest, SolveConvDiff256_Double_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_dbl;
        SORSolve<double> SOR_solve_d(A, b, *w, success_args);
        SOR_solve_d.solve();
        if (*show_plots) { SOR_solve_d.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_d.check_converged());
        EXPECT_LE(SOR_solve_d.get_relres(), conv_tol_dbl);

    }
    
}

TEST_F(SORTest, SolveConvDiff64_Single) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_sgl;
        SORSolve<float> SOR_solve_s(A, b, *w, success_args);
        SOR_solve_s.solve();
        if (*show_plots) { SOR_solve_s.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_s.check_converged());
        EXPECT_LE(SOR_solve_s.get_relres(), conv_tol_sgl);

    }

}

TEST_F(SORTest, SolveConvDiff256_Single_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_sgl;
        SORSolve<float> SOR_solve_s(A, b, *w, success_args);
        SOR_solve_s.solve();
        if (*show_plots) { SOR_solve_s.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_s.check_converged());
        EXPECT_LE(SOR_solve_s.get_relres(), conv_tol_sgl);

    }

}

TEST_F(SORTest, SolveConvDiff64_SingleFailBeyondEpsilon) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        fail_args.target_rel_res = 0.1*u_sgl;
        SORSolve<float> SOR_solve_s(A, b, *w, fail_args);
        SOR_solve_s.solve();
        if (*show_plots) { SOR_solve_s.view_relres_plot("log"); }
    
        EXPECT_FALSE(SOR_solve_s.check_converged());
        EXPECT_GT(SOR_solve_s.get_relres(), 0.1*u_sgl);

    }

}

TEST_F(SORTest, SolveConvDiff64_Half) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_hlf;
        SORSolve<half> SOR_solve_h(A, b, *w, success_args);
        SOR_solve_h.solve();
        if (*show_plots) { SOR_solve_h.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_h.check_converged());
        EXPECT_LE(SOR_solve_h.get_relres(), conv_tol_hlf);

    }

}

TEST_F(SORTest, SolveConvDiff256_Half_LONGRUNTIME) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_256_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;

        success_args.target_rel_res = conv_tol_hlf;
        SORSolve<half> SOR_solve_h(A, b, *w, success_args);
        SOR_solve_h.solve();
        if (*show_plots) { SOR_solve_h.view_relres_plot("log"); }
        
        EXPECT_TRUE(SOR_solve_h.check_converged());
        EXPECT_LE(SOR_solve_h.get_relres(), conv_tol_hlf);

    }

}

TEST_F(SORTest, SolveConvDiff64_HalfFailBeyondEpsilon) {
    
    Matrix<double, Dynamic, Dynamic> A(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_A.csv"));
    Matrix<double, Dynamic, 1> b(read_matrix_csv<double>(solve_matrix_dir + "conv_diff_64_b.csv"));

    for (auto w = ws.cbegin(); w != ws.cend(); ++w) {

        cout << "Testing w=" << *w << endl;
        
        fail_args.target_rel_res = 0.1*u_hlf;
        SORSolve<half> SOR_solve_h(A, b, *w, fail_args);
        SOR_solve_h.solve();
        if (*show_plots) { SOR_solve_h.view_relres_plot("log"); }
    
        EXPECT_FALSE(SOR_solve_h.check_converged());
        EXPECT_GT(SOR_solve_h.get_relres(), 0.1*u_hlf);

    }

}

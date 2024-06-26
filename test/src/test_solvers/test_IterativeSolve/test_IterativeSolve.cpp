#include "test.h"

#include "test_IterativeSolve.h"

class TypedIterativeSolve_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestConstructors() {

        // Test with no initial guess and default parameters
        constexpr int n(6);
        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, n, n
            )
        );
        Vector<double> b(
            Vector<double>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> soln(
            Vector<TPrecision>::Random(TestBase::bundle, 1)
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        TypedIterativeSolveTestingMock<TMatrix, TPrecision> test_mock_no_guess(
            &typed_lin_sys, soln, default_args
        );

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess,
            Vector<double>::Ones(TestBase::bundle, n)
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.get_generic_soln(),
            Vector<double>::Ones(TestBase::bundle, n)
        );

        ASSERT_VECTOR_EQ(
            test_mock_no_guess.init_guess_typed,
            Vector<double>::Ones(
                TestBase::bundle, n
            ).template cast<TPrecision>()
        );
        ASSERT_VECTOR_EQ(
            test_mock_no_guess.get_typed_soln(),
            Vector<double>::Ones(
                TestBase::bundle, n
            ).template cast<TPrecision>()
        );

        EXPECT_EQ(test_mock_no_guess.max_iter, 100);
        EXPECT_EQ(test_mock_no_guess.target_rel_res, std::pow(10, -10));

        EXPECT_FALSE(test_mock_no_guess.check_initiated());
        EXPECT_FALSE(test_mock_no_guess.check_converged());
        EXPECT_FALSE(test_mock_no_guess.check_terminated());
        EXPECT_EQ(test_mock_no_guess.get_iteration(), 0);
        EXPECT_EQ(test_mock_no_guess.get_res_norm_history().size(), 1);
        EXPECT_NEAR(
            test_mock_no_guess.get_res_norm_history()[0],
            (b-A*test_mock_no_guess.init_guess).norm().get_scalar(),
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_EQ(test_mock_no_guess.get_res_costheta_history().size(), 1);
        EXPECT_NEAR(
            test_mock_no_guess.get_res_costheta_history()[0],
            0.,
            Tol<TPrecision>::gamma(n)
        );

        // Test with initial guess and explicit parameters
        Vector<double> init_guess(Vector<double>::Random(TestBase::bundle, n));
        SolveArgPkg args;
        args.init_guess = init_guess;
        args.max_iter = n;
        args.target_rel_res = std::pow(10, -4);

        TypedIterativeSolveTestingMock<TMatrix, TPrecision> test_mock_guess(
            &typed_lin_sys, soln, args
        );

        ASSERT_VECTOR_EQ(test_mock_guess.init_guess, init_guess);
        ASSERT_VECTOR_EQ(
            test_mock_guess.get_generic_soln(),
            init_guess.template cast<TPrecision>().template cast<double>()
        );

        ASSERT_VECTOR_EQ(
            test_mock_guess.init_guess_typed,
            init_guess.template cast<TPrecision>()
        );
        ASSERT_VECTOR_EQ(
            test_mock_guess.get_typed_soln(),
            init_guess.template cast<TPrecision>()
        );

        EXPECT_EQ(test_mock_guess.max_iter, args.max_iter);
        EXPECT_EQ(test_mock_guess.target_rel_res, args.target_rel_res);

        EXPECT_FALSE(test_mock_guess.check_initiated());
        EXPECT_FALSE(test_mock_guess.check_converged());
        EXPECT_FALSE(test_mock_guess.check_terminated());
        EXPECT_EQ(test_mock_guess.get_iteration(), 0);
        EXPECT_EQ(test_mock_guess.get_res_norm_history().size(), 1);
        EXPECT_NEAR(
            test_mock_guess.get_res_norm_history()[0],
            (b - A*init_guess).norm().get_scalar(),
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_EQ(test_mock_guess.get_res_costheta_history().size(), 1);
        EXPECT_NEAR(
            test_mock_guess.get_res_costheta_history()[0],
            0.,
            Tol<TPrecision>::gamma(n)
        );

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestSolve() {

        constexpr int n(64);
        constexpr int max_iter(5);
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_A.csv")
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_b.csv")
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        SolveArgPkg args;
        Vector<TPrecision> typed_soln(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_x.csv")
        ));
        Vector<double> init_guess(Vector<double>::Ones(TestBase::bundle, n));
        args.init_guess = init_guess;
        args.max_iter = max_iter;
        args.target_rel_res = (
            Tol<TPrecision>::roundoff() +
            ((b-A*typed_soln.template cast<double>()).norm() /
             (b-A*init_guess).norm()).get_scalar()
        );

        TypedIterativeSolveTestingMock<TMatrix, TPrecision> test_mock(
            &typed_lin_sys, typed_soln, args
        );

        EXPECT_NEAR(test_mock.get_relres(), 1., Tol<TPrecision>::gamma(n));
    
        test_mock.solve();

        // Check init_guess doesn't change
        ASSERT_VECTOR_EQ(test_mock.init_guess, init_guess);
        ASSERT_VECTOR_EQ(
            test_mock.init_guess_typed,
            init_guess.template cast<TPrecision>()
        );

        // Check changed soln on iterate
        ASSERT_VECTOR_EQ(test_mock.get_typed_soln(), typed_soln);

        // Check convergence
        EXPECT_TRUE(test_mock.check_initiated());
        EXPECT_TRUE(test_mock.check_converged());
        EXPECT_TRUE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.get_iteration(), 1);

        // Check residual history and relres correctly calculates
        EXPECT_EQ(test_mock.get_res_norm_history().size(), 2);
        EXPECT_NEAR(
            test_mock.get_res_norm_history()[0],
            (b-A*init_guess).norm().get_scalar(),
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.get_res_norm_history()[1],
            (b-A*(typed_soln.template cast<double>())).norm().get_scalar(),
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.get_relres(),
            ((b-A*(typed_soln.template cast<double>())).norm() /
             (b-A*init_guess).norm()).get_scalar(),
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_EQ(test_mock.get_res_costheta_history().size(), 2);
        EXPECT_NEAR(
            test_mock.get_res_costheta_history()[0],
            0.,
            Tol<TPrecision>::gamma(n)
        );
        EXPECT_NEAR(
            test_mock.get_res_costheta_history()[1],
            ((b -
              A*(typed_soln.template cast<double>())).dot(b-A*init_guess) /
             ((b - A*(typed_soln.template cast<double>())).norm() *
              (b - A*init_guess).norm())
            ).get_scalar(),
            Tol<TPrecision>::gamma(n)
        );

        if (*show_plots) { test_mock.view_relres_plot(); }

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestReset() {

        constexpr int n(64);
        TMatrix<double> A(read_matrixCSV<TMatrix, double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_A.csv")
        ));
        Vector<double> b(read_vectorCSV<double>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_b.csv")
        ));
        Vector<TPrecision> typed_soln(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("conv_diff_64_x.csv")
        ));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        TypedIterativeSolveTestingMock<TMatrix, TPrecision> test_mock(
            &typed_lin_sys, typed_soln, default_args
        );

        // Call solve and then reset
        test_mock.solve();
        test_mock.reset();

        // Check init_guess doesn't change
        ASSERT_VECTOR_EQ(
            test_mock.init_guess,
            Vector<double>::Ones(TestBase::bundle, n)
        );

        // Check solve variables are all reset
        ASSERT_VECTOR_EQ(
            test_mock.get_typed_soln(),
            Vector<TPrecision>::Ones(TestBase::bundle, n)
        );
        EXPECT_FALSE(test_mock.check_initiated());
        EXPECT_FALSE(test_mock.check_converged());
        EXPECT_FALSE(test_mock.check_terminated());
        EXPECT_EQ(test_mock.get_iteration(), 0);
        std::vector<double> init_res_norm_history {
            (b - A*test_mock.init_guess).norm().get_scalar()
        };
        EXPECT_EQ(test_mock.get_res_norm_history(), init_res_norm_history);
        std::vector<double> init_res_costheta_history {0.};
        EXPECT_EQ(
            test_mock.get_res_costheta_history(),
            init_res_costheta_history
        );

    }

    template <template <typename> typename TMatrix>
    void TestMismatchedCols() {

        auto try_create_solve_mismatched_cols = []() {

            SolveArgPkg args;
            args.init_guess = Vector<double>::Ones(TestBase::bundle, 5, 1);

            GenericLinearSystem<TMatrix> gen_lin_sys(
                TMatrix<double>::Ones(TestBase::bundle, 64, 64),
                Vector<double>::Ones(TestBase::bundle, 64)
            );
            TypedLinearSystem<TMatrix, double> typed_lin_sys(&gen_lin_sys);

            TypedIterativeSolveTestingMock<TMatrix, double> test(
                &typed_lin_sys, Vector<double>::Ones(TestBase::bundle, 64), args
            );

        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            try_create_solve_mismatched_cols
        );

    }

    template <template <typename> typename TMatrix>
    void TestErrorNonSquare() {

        auto try_create_solve_non_square = [this]() {

            GenericLinearSystem<TMatrix> gen_lin_sys(
                TMatrix<double>::Ones(TestBase::bundle, 43, 64),
                Vector<double>::Ones(TestBase::bundle, 42)
            );
            TypedLinearSystem<TMatrix, double> typed_lin_sys(&gen_lin_sys);

            TypedIterativeSolveTestingMock<TMatrix, double> test(
                &typed_lin_sys, Vector<double>::Ones(TestBase::bundle, 64),
                default_args
            );
        };

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_create_solve_non_square);

    }

};

TEST_F(TypedIterativeSolve_Test, TestConstructors_Half_SOLVER) {
    TestConstructors<MatrixDense, half>();
    TestConstructors<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructors_Single_SOLVER) {
    TestConstructors<MatrixDense, float>();
    TestConstructors<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestConstructors_Double_SOLVER) {
    TestConstructors<MatrixDense, double>();
    TestConstructors<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelres_Half_SOLVER) {
    TestSolve<MatrixDense, half>();
    TestSolve<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelres_Single_SOLVER) {
    TestSolve<MatrixDense, float>();
    TestSolve<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestSolveAndRelres_Double_SOLVER) {
    TestSolve<MatrixDense, double>();
    TestSolve<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestReset_Half_SOLVER) {
    TestReset<MatrixDense, half>();
    TestReset<NoFillMatrixSparse, half>();
}

TEST_F(TypedIterativeSolve_Test, TestReset_Single_SOLVER) {
    TestReset<MatrixDense, float>();
    TestReset<NoFillMatrixSparse, float>();
}

TEST_F(TypedIterativeSolve_Test, TestReset_Double_SOLVER) {
    TestReset<MatrixDense, double>();
    TestReset<NoFillMatrixSparse, double>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorMismatchedCols_SOLVER) {
    TestMismatchedCols<MatrixDense>();
    TestMismatchedCols<NoFillMatrixSparse>();
}

TEST_F(TypedIterativeSolve_Test, TestErrorNonSquare_SOLVER) {
    TestErrorNonSquare<MatrixDense>();
    TestErrorNonSquare<NoFillMatrixSparse>();
}
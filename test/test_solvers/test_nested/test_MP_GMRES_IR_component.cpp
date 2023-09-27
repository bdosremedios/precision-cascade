#include "test_MP_GMRES_IR_component.h"

class MP_GMRES_IR_ComponentTest: public TestBase
{
public:

    template <template <typename> typename M>
    void TestOuterIterateCorrectSolvers() {

        MP_GMRES_IR_Solve_TestingMock<M> test_mock(
            M<double>(2, 2), MatrixVector<double>(2, 1), default_args
        );

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::HLF_PHASE;
        test_mock.outer_iterate_setup();

        GMRESSolve<M, half> type_test_half(
            M<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::SGL_PHASE;
        test_mock.outer_iterate_setup();

        GMRESSolve<M, float> type_test_single(
            M<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::DBL_PHASE;
        test_mock.outer_iterate_setup();

        GMRESSolve<M, double> type_test_double(
            M<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_double));

    }

    template <template <typename> typename M>
    void TestReset() {

        // Check initial half set to float and test if reset to half
        MP_GMRES_IR_Solve_TestingMock<M> test_mock(
            M<double>(2, 2), MatrixVector<double>(2, 1), default_args
        );

        GMRESSolve<M, half> type_test_half(
            M<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::SGL_PHASE;
        test_mock.outer_iterate_setup();

        GMRESSolve<M, float> type_test_single(
            M<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.reset();
        EXPECT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

    }

};

TEST_F(MP_GMRES_IR_ComponentTest, Test_Constructor_Both) {

    MP_GMRES_IR_Solve_TestingMock<MatrixDense> dense_mock(
        MatrixDense<double>(2, 2), MatrixVector<double>(2, 1), default_args 
    );
    EXPECT_EQ(dense_mock.cascade_phase, MP_GMRES_IR_Solve_TestingMock<MatrixDense>::INIT_PHASE);

    MP_GMRES_IR_Solve_TestingMock<MatrixSparse> sparse_mock(
        MatrixSparse<double>(2, 2), MatrixVector<double>(2, 1), default_args
    );
    EXPECT_EQ(sparse_mock.cascade_phase, MP_GMRES_IR_Solve_TestingMock<MatrixSparse>::INIT_PHASE);

}

TEST_F(MP_GMRES_IR_ComponentTest, Test_SetCorrectPhaseSolvers_Both) {
    TestOuterIterateCorrectSolvers<MatrixDense>();
    TestOuterIterateCorrectSolvers<MatrixSparse>();
}

TEST_F(MP_GMRES_IR_ComponentTest, Test_Reset_Both) {
    TestReset<MatrixDense>();
    TestReset<MatrixSparse>();
}
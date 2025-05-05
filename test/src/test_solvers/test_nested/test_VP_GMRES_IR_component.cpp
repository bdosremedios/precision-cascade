#include "test_VP_GMRES_IR_component.h"

#include <cuda_fp16.h>

class VP_GMRES_IR_Component_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix>
    void TestConstructor() {

        GenericLinearSystem<TMatrix> gen_lin_sys(
            TMatrix<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        VP_GMRES_IR_Solve_TestingMock<TMatrix> dense_mock(
            &gen_lin_sys, default_args
        );
        EXPECT_TRUE(dense_mock.is_in_half_phase());

    }

    template <template <typename> typename TMatrix>
    void TestOuterIterateCorrectSolvers() {

        GenericLinearSystem<TMatrix> gen_lin_sys(
            CommonMatRandomInterface<TMatrix ,double>::rand_matrix(
                TestBase::bundle, 2, 2
            ),
            Vector<double>::Random(TestBase::bundle, 2, 1)
        );
        VP_GMRES_IR_Solve_TestingMock<TMatrix> test_mock(
            &gen_lin_sys, default_args
        );

        test_mock.set_phase_to_use = (
            VP_GMRES_IR_Solve_TestingMock<TMatrix>::HLF_PHASE
        );
        test_mock.outer_iterate_setup();

        TypedLinearSystem<TMatrix, __half> lin_sys_typ_hlf(&gen_lin_sys);
        GMRESSolve<TMatrix, __half> type_test_half(
            &lin_sys_typ_hlf, default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = (
            VP_GMRES_IR_Solve_TestingMock<TMatrix>::SGL_PHASE
        );
        test_mock.outer_iterate_setup();

        TypedLinearSystem<TMatrix, float> lin_sys_typ_sgl(&gen_lin_sys);
        GMRESSolve<TMatrix, float> type_test_single(
            &lin_sys_typ_sgl, default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.set_phase_to_use = (
            VP_GMRES_IR_Solve_TestingMock<TMatrix>::DBL_PHASE
        );
        test_mock.outer_iterate_setup();

        TypedLinearSystem<TMatrix, double> lin_sys_typ_dbl(&gen_lin_sys);
        GMRESSolve<TMatrix, double> type_test_double(
            &lin_sys_typ_dbl, default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_double));

    }

    template <template <typename> typename TMatrix>
    void TestReset() {

        // Check initial __half set to float and test if reset to __half
        GenericLinearSystem<TMatrix> gen_lin_sys(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, 2, 2
            ),
            Vector<double>::Random(TestBase::bundle, 2, 1)
        );
        VP_GMRES_IR_Solve_TestingMock<TMatrix> test_mock(
            &gen_lin_sys, default_args
        );

        TypedLinearSystem<TMatrix, __half> lin_sys_typ_hlf(&gen_lin_sys);
        GMRESSolve<TMatrix, __half> type_test_half(
            &lin_sys_typ_hlf, default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = (
            VP_GMRES_IR_Solve_TestingMock<TMatrix>::SGL_PHASE
        );
        test_mock.outer_iterate_setup();

        TypedLinearSystem<TMatrix, float> lin_sys_typ_sgl(&gen_lin_sys);
        GMRESSolve<TMatrix, float> type_test_single(
            &lin_sys_typ_sgl, default_args
        );
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.reset();
        EXPECT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

    }

};

TEST_F(VP_GMRES_IR_Component_Test, Test_Constructor_SOLVER) {
    TestConstructor<MatrixDense>();
    TestConstructor<NoFillMatrixSparse>();
}

TEST_F(VP_GMRES_IR_Component_Test, Test_SetCorrectPhaseSolvers_SOLVER) {
    TestOuterIterateCorrectSolvers<MatrixDense>();
    TestOuterIterateCorrectSolvers<NoFillMatrixSparse>();
}

TEST_F(VP_GMRES_IR_Component_Test, Test_Reset_SOLVER) {
    TestReset<MatrixDense>();
    TestReset<NoFillMatrixSparse>();
}
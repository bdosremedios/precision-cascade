#include <cuda_fp16.h>

#include "test_MP_GMRES_IR_component.h"

class MP_GMRES_IR_ComponentTest: public TestBase
{
public:

    template <template <typename> typename M>
    void TestConstructor() {

        GenericLinearSystem<M> lin_sys_dense(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        MP_GMRES_IR_Solve_TestingMock<M> dense_mock(lin_sys_dense, default_args);
        EXPECT_EQ(dense_mock.cascade_phase, MP_GMRES_IR_Solve_TestingMock<M>::INIT_PHASE);

    }

    template <template <typename> typename M>
    void TestOuterIterateCorrectSolvers() {

        GenericLinearSystem<M> lin_sys(
            CommonMatRandomInterface<M ,double>::rand_matrix(TestBase::bundle, 2, 2),
            Vector<double>::Random(TestBase::bundle, 2, 1)
        );
        MP_GMRES_IR_Solve_TestingMock<M> test_mock(lin_sys, default_args);

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::HLF_PHASE;
        test_mock.outer_iterate_setup();

        TypedLinearSystem<M, __half> lin_sys_typ_hlf(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        GMRESSolve<M, __half> type_test_half(lin_sys_typ_hlf, 1., default_args);
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::SGL_PHASE;
        test_mock.outer_iterate_setup();

        TypedLinearSystem<M, float> lin_sys_typ_sgl(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        GMRESSolve<M, float> type_test_single(lin_sys_typ_sgl, 1., default_args);
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::DBL_PHASE;
        test_mock.outer_iterate_setup();

        TypedLinearSystem<M, double> lin_sys_typ_dbl(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        GMRESSolve<M, double> type_test_double(lin_sys_typ_dbl, 1., default_args);
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_double));

    }

    template <template <typename> typename M>
    void TestReset() {

        // Check initial __half set to float and test if reset to __half
        GenericLinearSystem<M> lin_sys(
            CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 2, 2),
            Vector<double>::Random(TestBase::bundle, 2, 1)
        );
        MP_GMRES_IR_Solve_TestingMock<M> test_mock(lin_sys, default_args);

        TypedLinearSystem<M, __half> lin_sys_typ_hlf(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        GMRESSolve<M, __half> type_test_half(lin_sys_typ_hlf, 1., default_args);
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

        test_mock.set_phase_to_use = MP_GMRES_IR_Solve_TestingMock<M>::SGL_PHASE;
        test_mock.outer_iterate_setup();

        TypedLinearSystem<M, float> lin_sys_typ_sgl(
            M<double>(TestBase::bundle, 2, 2),
            Vector<double>(TestBase::bundle, 2, 1)
        );
        GMRESSolve<M, float> type_test_single(lin_sys_typ_sgl, 1., default_args);
        ASSERT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_single));

        test_mock.reset();
        EXPECT_EQ(typeid(*test_mock.inner_solver), typeid(type_test_half));

    }

};

TEST_F(MP_GMRES_IR_ComponentTest, Test_Constructor) {
    TestConstructor<MatrixDense>();
    TestConstructor<NoFillMatrixSparse>();
}

TEST_F(MP_GMRES_IR_ComponentTest, Test_SetCorrectPhaseSolvers) {
    TestOuterIterateCorrectSolvers<MatrixDense>();
    TestOuterIterateCorrectSolvers<NoFillMatrixSparse>();
}

TEST_F(MP_GMRES_IR_ComponentTest, Test_Reset) {
    TestReset<MatrixDense>();
    TestReset<NoFillMatrixSparse>();
}
#include "test_MP_GMRES_IR_component.h"

class MP_GMRES_IR_ComponentTest: public TestBase {};

TEST_F(MP_GMRES_IR_ComponentTest, Test_Constructor_Both) {

    MP_GMRES_IR_Solve_TestingMock<MatrixDense> dense_mock(
        MatrixDense<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args 
    );
    EXPECT_EQ(dense_mock.cascade_phase, MP_GMRES_IR_Solve_TestingMock<MatrixDense>::INIT_PHASE);

    MP_GMRES_IR_Solve_TestingMock<MatrixSparse> sparse_mock(
        MatrixSparse<double>(2, 2), MatrixVector<double>(2, 1), 1., default_args
    );
    EXPECT_EQ(sparse_mock.cascade_phase, MP_GMRES_IR_Solve_TestingMock<MatrixSparse>::INIT_PHASE);

}
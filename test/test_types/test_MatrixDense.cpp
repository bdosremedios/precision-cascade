#include "test_Matrix.h"

#include "types/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixDense, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixDense>(); }

};

TEST_F(MatrixDense_Test, TestListInitialization_Hlf) { TestListInitialization<half>(); }
TEST_F(MatrixDense_Test, TestListInitialization_Sgl) { TestListInitialization<float>(); }
TEST_F(MatrixDense_Test, TestListInitialization_Dbl) { TestListInitialization<double>(); }


TEST_F(MatrixDense_Test, TestBadListInitialization) { TestBadListInitialization(); }
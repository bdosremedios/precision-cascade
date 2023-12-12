#include "test_Matrix.h"

#include "types/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestConstruction() { TestConstruction_Base<MatrixDense, T>(); }

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixDense, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixDense>(); }

    template <typename T>
    void TestCoeffAccess() { TestCoeffAccess_Base<MatrixDense, T>(); }

    template <typename T>
    void TestStaticCreation() { TestStaticCreation_Base<MatrixDense, T>(); }

    template <typename T>
    void TestCol() { TestCol_Base<MatrixDense, T>(); }

};

TEST_F(MatrixDense_Test, TestConstruction) {
    TestConstruction<half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(MatrixDense_Test, TestListInitialization) {
    TestListInitialization<half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixDense_Test, TestBadListInitialization) { TestBadListInitialization(); }

TEST_F(MatrixDense_Test, TestCoeffAccess) {
    TestCoeffAccess<half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixDense_Test, TestStaticCreation) {
    TestStaticCreation<half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixDense_Test, TestCol) {
    TestCol<half>();
    TestCol<float>();
    TestCol<double>();
}

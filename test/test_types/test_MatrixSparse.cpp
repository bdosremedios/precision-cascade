#include "test_Matrix.h"

#include "types/MatrixSparse.h"

class MatrixSparse_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestConstruction() { TestConstruction_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixSparse, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixSparse>(); }

    template <typename T>
    void TestCoeffAccess() { TestCoeffAccess_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestStaticCreation() { TestStaticCreation_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestCol() { TestCol_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestBlock() { TestBlock_Base<MatrixSparse, T>(); }

};

TEST_F(MatrixSparse_Test, TestConstruction) {
    TestConstruction<half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(MatrixSparse_Test, TestListInitialization) {
    TestListInitialization<half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixSparse_Test, TestBadListInitialization) { TestBadListInitialization(); }

TEST_F(MatrixSparse_Test, TestCoeffAccess) {
    TestCoeffAccess<half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixSparse_Test, TestStaticCreation) {
    TestStaticCreation<half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixSparse_Test, TestCol) {
    TestCol<half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(MatrixSparse_Test, TestBlock) {
    TestBlock<half>();
    TestBlock<float>();
    TestBlock<double>();
}

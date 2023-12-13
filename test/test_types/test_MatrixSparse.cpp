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
    void TestTranspose() { TestTranspose_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestScale() { TestScale_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestMatVec() { TestMatVec_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestMatMat() { TestMatMat_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestNorm() { TestNorm_Base<MatrixSparse, T>(); }

    template <typename T>
    void TestAddSub() { TestAddSub_Base<MatrixSparse, T>(); }

    void TestCast() { TestCast_Base<MatrixSparse>(); }

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

TEST_F(MatrixSparse_Test, TestTranspose) {
    TestTranspose<half>();
    TestTranspose<float>();
    TestTranspose<double>();
}

TEST_F(MatrixSparse_Test, TestScale) {
    TestScale<half>();
    TestScale<float>();
    TestScale<double>();
}

TEST_F(MatrixSparse_Test, TestMatVec) {
    TestMatVec<half>();
    TestMatVec<float>();
    TestMatVec<double>();
}

TEST_F(MatrixSparse_Test, TestMatMat) {
    TestMatMat<half>();
    TestMatMat<float>();
    TestMatMat<double>();
}

TEST_F(MatrixSparse_Test, TestNorm) {
    TestNorm<half>();
    TestNorm<float>();
    TestNorm<double>();
}

TEST_F(MatrixSparse_Test, TestAddSub) {
    TestAddSub<half>();
    TestAddSub<float>();
    TestAddSub<double>();
}

TEST_F(MatrixSparse_Test, TestCast) { TestCast(); }
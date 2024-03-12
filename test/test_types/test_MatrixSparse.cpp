#include "test_Matrix.h"

#include "types/MatrixSparse/MatrixSparse.h"

class MatrixSparse_Test: public Matrix_Test<MatrixSparse>
{
public:

    template <typename T>
    void TestCoeffAccess() { TestCoeffAccess_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestPropertyAccess() { TestPropertyAccess_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestConstruction() { TestConstruction_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestListInitialization() { TestListInitialization_Base<MatrixSparse, T>(); }

    // void TestBadListInitialization() { TestBadListInitialization_Base<MatrixSparse>(); }

    // template <typename T>
    // void TestCopyAssignment() { TestCopyAssignment_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestCopyConstructor() { TestCopyConstructor_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestStaticCreation() { TestStaticCreation_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestCol() { TestCol_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestTranspose() { TestTranspose_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestScale() { TestScale_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatVec() { TestMatVec_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatMat() { TestMatMat_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestNorm() { TestNorm_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestAddSub() { TestAddSub_Base<MatrixSparse, T>(); }

    // void TestCast() { TestCast_Base<MatrixSparse>(); }

};

TEST_F(MatrixSparse_Test, TestCoeffAccess) {
    TestCoeffAccess<__half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

// TEST_F(MatrixSparse_Test, TestBadCoeffAccess) { TestBadCoeffAccess(); }

// TEST_F(MatrixSparse_Test, TestPropertyAccess) {
//     TestPropertyAccess<__half>();
//     TestPropertyAccess<float>();
//     TestPropertyAccess<double>();
// }

// TEST_F(MatrixSparse_Test, TestConstruction) {
//     TestConstruction<__half>();
//     TestConstruction<float>();
//     TestConstruction<double>();
// }

// TEST_F(MatrixSparse_Test, TestListInitialization) {
//     TestListInitialization<__half>();
//     TestListInitialization<float>();
//     TestListInitialization<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadListInitialization) { TestBadListInitialization(); }

// TEST_F(MatrixSparse_Test, TestNonZeros) {
//     TestNonZeros<__half>();
//     TestNonZeros<float>();
//     TestNonZeros<double>();
// }

// TEST_F(MatrixSparse_Test, TestDynamicMemConstruction) {
//     TestDynamicMemConstruction<__half>();
//     TestDynamicMemConstruction<float>();
//     TestDynamicMemConstruction<double>();
// }

// TEST_F(MatrixSparse_Test, TestDynamicMemCopyToPtr) {
//     TestDynamicMemCopyToPtr<__half>();
//     TestDynamicMemCopyToPtr<float>();
//     TestDynamicMemCopyToPtr<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadDynamicMemCopyToPtr) { TestBadDynamicMemCopyToPtr(); }

// TEST_F(MatrixSparse_Test, TestCopyAssignment) {
//     TestCopyAssignment<__half>();
//     TestCopyAssignment<float>();
//     TestCopyAssignment<double>();
// }

// TEST_F(MatrixSparse_Test, TestCopyConstructor) {
//     TestCopyConstructor<__half>();
//     TestCopyConstructor<float>();
//     TestCopyConstructor<double>();
// }

// TEST_F(MatrixSparse_Test, TestStaticCreation) {
//     TestStaticCreation<__half>();
//     TestStaticCreation<float>();
//     TestStaticCreation<double>();
// }

// TEST_F(MatrixSparse_Test, TestCol) {
//     TestCol<__half>();
//     TestCol<float>();
//     TestCol<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadCol) { TestBadCol(); }

// TEST_F(MatrixSparse_Test, TestBlock) {
//     TestBlock<__half>();
//     TestBlock<float>();
//     TestBlock<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadBlock) { TestBadBlock(); }

// TEST_F(MatrixSparse_Test, TestScale) {
//     TestScale<__half>();
//     TestScale<float>();
//     TestScale<double>();
// }

// TEST_F(MatrixSparse_Test, TestScaleAssignment) {
//     TestScaleAssignment<__half>();
//     TestScaleAssignment<float>();
//     TestScaleAssignment<double>();
// }

// TEST_F(MatrixSparse_Test, TestMaxMagElem) {
//     TestMaxMagElem<__half>();
//     TestMaxMagElem<float>();
//     TestMaxMagElem<double>();
// }

// TEST_F(MatrixSparse_Test, TestNormalizeMagnitude) {
//     TestNormalizeMagnitude<__half>();
//     TestNormalizeMagnitude<float>();
//     TestNormalizeMagnitude<double>();
// }

// TEST_F(MatrixSparse_Test, TestMatVec) {
//     TestMatVec<__half>();
//     TestMatVec<float>();
//     TestMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadMatVec) {
//     TestBadMatVec<__half>();
//     TestBadMatVec<float>();
//     TestBadMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestTransposeMatVec) {
//     TestTransposeMatVec<__half>();
//     TestTransposeMatVec<float>();
//     TestTransposeMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadTransposeMatVec) {
//     TestBadTransposeMatVec<__half>();
//     TestBadTransposeMatVec<float>();
//     TestBadTransposeMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestTranspose) {
//     TestTranspose<__half>();
//     TestTranspose<float>();
//     TestTranspose<double>();
// }

// TEST_F(MatrixSparse_Test, TestMatMat) {
//     TestMatMat<__half>();
//     TestMatMat<float>();
//     TestMatMat<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadMatMat) {
//     TestBadMatMat<__half>();
//     TestBadMatMat<float>();
//     TestBadMatMat<double>();
// }

// TEST_F(MatrixSparse_Test, TestAddSub) {
//     TestAddSub<__half>();
//     TestAddSub<float>();
//     TestAddSub<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadAddSub) {
//     TestBadAddSub<__half>();
//     TestBadAddSub<float>();
//     TestBadAddSub<double>();
// }

// TEST_F(MatrixSparse_Test, TestNorm) {
//     TestNorm<__half>();
//     TestNorm<float>();
//     TestNorm<double>();
// }

// TEST_F(MatrixSparse_Test, TestCast) { TestCast(); }

// TEST_F(MatrixSparse_Test, TestBadCast) { TestBadCast(); }

// class MatrixSparse_Substitution_Test: public Matrix_Substitution_Test<MatrixSparse> {};

// TEST_F(MatrixSparse_Substitution_Test, TestBackwardSubstitution) {
//     TestBackwardSubstitution<__half>();
//     TestBackwardSubstitution<float>();
//     TestBackwardSubstitution<double>();
// }

// TEST_F(MatrixSparse_Substitution_Test, TestForwardSubstitution) {
//     TestForwardSubstitution<__half>();
//     TestForwardSubstitution<float>();
//     TestForwardSubstitution<double>();
// }
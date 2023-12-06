#include "test_Matrix.h"

#include "types/MatrixSparse.h"

class MatrixSparse_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixSparse, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixSparse>(); }

};

TEST_F(MatrixSparse_Test, TestListInitialization_Hlf) { TestListInitialization<half>(); }
TEST_F(MatrixSparse_Test, TestListInitialization_Sgl) { TestListInitialization<float>(); }
TEST_F(MatrixSparse_Test, TestListInitialization_Dbl) { TestListInitialization<double>(); }

TEST_F(MatrixSparse_Test, TestBadListInitialization) { TestBadListInitialization(); }
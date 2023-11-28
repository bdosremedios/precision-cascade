#include "../test.h"

// General matrix read tests
class MatrixRead_General_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadEmptyMatrix() {

        fs::path empty_file = read_matrix_dir / fs::path("empty.csv");
        M<double> test_empty(read_matrixCSV<M, double>(empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <template <typename> typename M>
    void ReadBadMatrices() {

        // Try to load non-existent file
        fs::path bad_file_0 = read_matrix_dir / fs::path("thisfile");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_0));
            FAIL();
        } catch (runtime_error e) { ; }

        // Try to load file with too small row
        fs::path bad_file_1 = read_matrix_dir / fs::path("bad1.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_1));
            FAIL();
        } catch (runtime_error e) { ;  }

        // Try to load file with too big rows
        fs::path bad_file_2 = read_matrix_dir / fs::path("bad2.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_2));
            FAIL();
        } catch (runtime_error e) { ; }

        // Try to load file with invalid character argument
        fs::path bad_file_3 = read_matrix_dir / fs::path("bad3.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_3));
            FAIL();
        } catch (runtime_error e) { ; }

    }

};

TEST_F(MatrixRead_General_Test, ReadEmptyMatrix_Both) {
    ReadEmptyMatrix<MatrixDense>();
    ReadEmptyMatrix<MatrixSparse>();
}

TEST_F(MatrixRead_General_Test, ReadBadFiles_Both) {
    ReadBadMatrices<MatrixDense>();
    ReadBadMatrices<MatrixSparse>();
}

template <typename T>
class MatrixRead_T_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadSquareMatrix(double u) {

        M<double> temp_target1(3, 3);
        temp_target1.coeffRef(0, 0) = 1; temp_target1.coeffRef(0, 1) = 2; temp_target1.coeffRef(0, 2) = 3;
        temp_target1.coeffRef(1, 0) = 4; temp_target1.coeffRef(1, 1) = 5; temp_target1.coeffRef(1, 2) = 6;
        temp_target1.coeffRef(2, 0) = 7; temp_target1.coeffRef(2, 1) = 8; temp_target1.coeffRef(2, 2) = 9;
        M<T> target1 = temp_target1.template cast<T>();

        M<double> temp_target2(5, 5);
        temp_target2.coeffRef(0, 0) = 1; temp_target2.coeffRef(0, 1) = 2; temp_target2.coeffRef(0, 2) = 3; temp_target2.coeffRef(0, 3) = 4; temp_target2.coeffRef(0, 4) = 5;
        temp_target2.coeffRef(1, 0) = 6; temp_target2.coeffRef(1, 1) = 7; temp_target2.coeffRef(1, 2) = 8; temp_target2.coeffRef(1, 3) = 9; temp_target2.coeffRef(1, 4) = 10;
        temp_target2.coeffRef(2, 0) = 11; temp_target2.coeffRef(2, 1) = 12; temp_target2.coeffRef(2, 2) = 13; temp_target2.coeffRef(2, 3) = 14; temp_target2.coeffRef(2, 4) = 15;
        temp_target2.coeffRef(3, 0) = 16; temp_target2.coeffRef(3, 1) = 17; temp_target2.coeffRef(3, 2) = 18; temp_target2.coeffRef(3, 3) = 19; temp_target2.coeffRef(3, 4) = 20;
        temp_target2.coeffRef(4, 0) = 21; temp_target2.coeffRef(4, 1) = 22; temp_target2.coeffRef(4, 2) = 23; temp_target2.coeffRef(4, 3) = 24; temp_target2.coeffRef(4, 4) = 25;
        M<T> target2 = temp_target2.template cast<T>();
    
        fs::path square1_file = read_matrix_dir / fs::path("square1.csv");
        fs::path square2_file = read_matrix_dir / fs::path("square2.csv");
        M<T> test1(read_matrixCSV<M, T>(square1_file));
        M<T> test2(read_matrixCSV<M, T>(square2_file));

        ASSERT_MATRIX_NEAR(test1, target1, static_cast<T>(u));
        ASSERT_MATRIX_NEAR(test2, target2, static_cast<T>(u));

    }

    template <template <typename> typename M>
    void ReadWideTallMatrix(double u) {

        M<double> temp_target_wide(2, 5);
        temp_target_wide.coeffRef(0, 0) = 10; temp_target_wide.coeffRef(0, 1) = 9; temp_target_wide.coeffRef(0, 2) = 8; temp_target_wide.coeffRef(0, 3) = 7; temp_target_wide.coeffRef(0, 4) = 6;
        temp_target_wide.coeffRef(1, 0) = 5; temp_target_wide.coeffRef(1, 1) = 4; temp_target_wide.coeffRef(1, 2) = 3; temp_target_wide.coeffRef(1, 3) = 2; temp_target_wide.coeffRef(1, 4) = 1;
        M<T> target_wide = temp_target_wide.template cast<T>();

        M<double> temp_target_tall(4, 2);
        temp_target_tall.coeffRef(0, 0) = 1; temp_target_tall.coeffRef(0, 1) = 2;
        temp_target_tall.coeffRef(1, 0) = 3; temp_target_tall.coeffRef(1, 1) = 4;
        temp_target_tall.coeffRef(2, 0) = 5; temp_target_tall.coeffRef(2, 1) = 6;
        temp_target_tall.coeffRef(3, 0) = 7; temp_target_tall.coeffRef(3, 1) = 8;
        M<T> target_tall = temp_target_tall.template cast<T>();

        fs::path wide_file = read_matrix_dir / fs::path("wide.csv");
        fs::path tall_file = read_matrix_dir / fs::path("tall.csv");

        M<T> test_wide(read_matrixCSV<M, T>(wide_file));
        M<T> test_tall(read_matrixCSV<M, T>(tall_file));

        ASSERT_MATRIX_NEAR(test_wide, target_wide, static_cast<T>(u));
        ASSERT_MATRIX_NEAR(test_tall, target_tall, static_cast<T>(u));

    }

    template <template <typename> typename M>
    void ReadPrecise(
        M<T> target_precise,
        fs::path precise_file,
        double u
    ) {

        M<T> test_precise(read_matrixCSV<M, T>(precise_file));
        ASSERT_MATRIX_NEAR(test_precise, target_precise, static_cast<T>(u));

    }

    template <template <typename> typename M>
    void ReadDifferentThanPrecise(
        M<T> target_precise,
        fs::path precise_file,
        double u
    ) {

        T eps = static_cast<T>(1.5*u);
        M<T> miss_precise_up = target_precise + M<T>::Ones(2, 2)*eps;
        M<T> miss_precise_down = target_precise - M<T>::Ones(2, 2)*eps;

        M<T> test_precise(read_matrixCSV<M, T>(precise_file));
        ASSERT_MATRIX_LT(test_precise, miss_precise_up);
        ASSERT_MATRIX_GT(test_precise, miss_precise_down);

    }

};

// All type vector read tests
class MatrixRead_Vector_Test: public TestBase
{
public:

    template <typename T>
    void ReadVector(double u) {

        MatrixVector<T> target(6);
        target(0) = 1; target(1) = 2; target(2) = 3; target(3) = 4; target(4) = 5; target(5) = 6;

        fs::path vector_file = read_matrix_dir / fs::path("vector.csv");
        MatrixVector<T> test(read_matrixCSV<MatrixVector, T>(vector_file));

        ASSERT_VECTOR_NEAR(test, target, static_cast<T>(u));

    }

};

TEST_F(MatrixRead_Vector_Test, ReadDoubleVector) { ReadVector<double>(u_dbl); }
TEST_F(MatrixRead_Vector_Test, ReadSingleVector) { ReadVector<double>(u_sgl); }
TEST_F(MatrixRead_Vector_Test, ReadHalfVector) { ReadVector<double>(u_hlf); }

TEST_F(MatrixRead_Vector_Test, FailOnMatrix) {    
    fs::path mat = read_matrix_dir / fs::path("square1.csv");
    try {
        MatrixVector<double> test(read_matrixCSV<MatrixVector, double>(mat));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }
}

// Double type matrix read tests
class MatrixRead_Double_Test: public MatrixRead_T_Test<double> {};

TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_dbl);}
TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_dbl); }
TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrix_Both) {

    MatrixDense<double> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = 1.12345678901232; target_precise_dense.coeffRef(0, 1) = 1.12345678901234;
    target_precise_dense.coeffRef(1, 0) = 1.12345678901236; target_precise_dense.coeffRef(1, 1) = 1.12345678901238;

    MatrixSparse<double> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = 1.12345678901232; target_precise_sparse.coeffRef(0, 1) = 1.12345678901234;
    target_precise_sparse.coeffRef(1, 0) = 1.12345678901236; target_precise_sparse.coeffRef(1, 1) = 1.12345678901238;

    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");

    ReadPrecise<MatrixDense>(target_precise_dense, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_dbl);

}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrix_Both) {

    MatrixDense<double> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = 1.12345678901232; target_precise_dense.coeffRef(0, 1) = 1.12345678901234;
    target_precise_dense.coeffRef(1, 0) = 1.12345678901236; target_precise_dense.coeffRef(1, 1) = 1.12345678901238;

    MatrixSparse<double> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = 1.12345678901232; target_precise_sparse.coeffRef(0, 1) = 1.12345678901234;
    target_precise_sparse.coeffRef(1, 0) = 1.12345678901236; target_precise_sparse.coeffRef(1, 1) = 1.12345678901238;

    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");

    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_dbl);

}

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrixDoubleLimit_Both) {

    MatrixDense<double> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = 1.1234567890123452; target_precise_dense.coeffRef(0, 1) = 1.1234567890123454;
    target_precise_dense.coeffRef(1, 0) = 1.1234567890123456; target_precise_dense.coeffRef(1, 1) = 1.1234567890123458;

    MatrixSparse<double> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = 1.1234567890123452; target_precise_sparse.coeffRef(0, 1) = 1.1234567890123454;
    target_precise_sparse.coeffRef(1, 0) = 1.1234567890123456; target_precise_sparse.coeffRef(1, 1) = 1.1234567890123458;

    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");

    ReadPrecise<MatrixDense>(target_precise_dense, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_dbl);

}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrixDoubleLimit) {

    MatrixDense<double> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = 1.1234567890123452; target_precise_dense.coeffRef(0, 1) = 1.1234567890123454;
    target_precise_dense.coeffRef(1, 0) = 1.1234567890123456; target_precise_dense.coeffRef(1, 1) = 1.1234567890123458;

    MatrixSparse<double> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = 1.1234567890123452; target_precise_sparse.coeffRef(0, 1) = 1.1234567890123454;
    target_precise_sparse.coeffRef(1, 0) = 1.1234567890123456; target_precise_sparse.coeffRef(1, 1) = 1.1234567890123458;

    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");

    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_dbl);

}

// Single type matrix read tests
class MatrixRead_Single_Test: public MatrixRead_T_Test<float> {};

TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_sgl);}
TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_sgl); }
TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixRead_Single_Test, ReadPreciseMatrix_Both) {

    MatrixDense<float> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = static_cast<float>(1.12345672); target_precise_dense.coeffRef(0, 1) = static_cast<float>(1.12345674);
    target_precise_dense.coeffRef(1, 0) = static_cast<float>(1.12345676); target_precise_dense.coeffRef(1, 1) = static_cast<float>(1.12345678);

    MatrixSparse<float> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = static_cast<float>(1.12345672); target_precise_sparse.coeffRef(0, 1) = static_cast<float>(1.12345674);
    target_precise_sparse.coeffRef(1, 0) = static_cast<float>(1.12345676); target_precise_sparse.coeffRef(1, 1) = static_cast<float>(1.12345678);

    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");
    ReadPrecise<MatrixDense>(target_precise_dense, precise_file, u_sgl);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_sgl);

}

TEST_F(MatrixRead_Single_Test, ReadDifferentThanPreciseMatrix_Both) {

    MatrixDense<float> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = static_cast<float>(1.12345672); target_precise_dense.coeffRef(0, 1) = static_cast<float>(1.12345674);
    target_precise_dense.coeffRef(1, 0) = static_cast<float>(1.12345676); target_precise_dense.coeffRef(1, 1) = static_cast<float>(1.12345678);

    MatrixSparse<float> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = static_cast<float>(1.12345672); target_precise_sparse.coeffRef(0, 1) = static_cast<float>(1.12345674);
    target_precise_sparse.coeffRef(1, 0) = static_cast<float>(1.12345676); target_precise_sparse.coeffRef(1, 1) = static_cast<float>(1.12345678);

    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");

    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file, u_sgl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_sgl);

}

// Half type matrix read tests
class MatrixRead_Half_Test: public MatrixRead_T_Test<half> {};

TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_hlf);}
TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_hlf); }
TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixRead_Half_Test, ReadPreciseMatrix) {

    MatrixDense<half> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = static_cast<half>(1.123); target_precise_dense.coeffRef(0, 1) = static_cast<half>(1.124);
    target_precise_dense.coeffRef(1, 0) = static_cast<half>(1.125); target_precise_dense.coeffRef(1, 1) = static_cast<half>(1.126);

    MatrixSparse<half> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = static_cast<half>(1.123); target_precise_sparse.coeffRef(0, 1) = static_cast<half>(1.124);
    target_precise_sparse.coeffRef(1, 0) = static_cast<half>(1.125); target_precise_sparse.coeffRef(1, 1) = static_cast<half>(1.126);

    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");

    ReadPrecise<MatrixDense>(target_precise_dense, precise_file, u_hlf);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_hlf);

}

TEST_F(MatrixRead_Half_Test, ReadDifferentThanPreciseMatrix) {

    MatrixDense<half> target_precise_dense(2, 2);
    target_precise_dense.coeffRef(0, 0) = static_cast<half>(1.123); target_precise_dense.coeffRef(0, 1) = static_cast<half>(1.124);
    target_precise_dense.coeffRef(1, 0) = static_cast<half>(1.125); target_precise_dense.coeffRef(1, 1) = static_cast<half>(1.126);

    MatrixSparse<half> target_precise_sparse(2, 2);
    target_precise_sparse.coeffRef(0, 0) = static_cast<half>(1.123); target_precise_sparse.coeffRef(0, 1) = static_cast<half>(1.124);
    target_precise_sparse.coeffRef(1, 0) = static_cast<half>(1.125); target_precise_sparse.coeffRef(1, 1) = static_cast<half>(1.126);

    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");

    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file, u_hlf);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file, u_hlf);

}

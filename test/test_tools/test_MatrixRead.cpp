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
    void ReadSquareMatrix() {

        M<double> temp_target1 ({
            {1,  2,  3},
            {4,  5,  6},
            {7,  8,  9}
        });
        M<T> target1 = temp_target1.template cast<T>();

        M<double> temp_target2 ({
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        });
        M<T> target2 = temp_target2.template cast<T>();
    
        fs::path square1_file = read_matrix_dir / fs::path("square1.csv");
        fs::path square2_file = read_matrix_dir / fs::path("square2.csv");
        M<T> test1(read_matrixCSV<M, T>(square1_file));
        M<T> test2(read_matrixCSV<M, T>(square2_file));

        ASSERT_MATRIX_NEAR(test1, target1, Tol<T>::roundoff_T());
        ASSERT_MATRIX_NEAR(test2, target2, Tol<T>::roundoff_T());

    }

    template <template <typename> typename M>
    void ReadWideTallMatrix() {

        M<double> temp_target_wide ({
            {10, 9, 8, 7, 6},
            {5, 4, 3, 2, 1}
        });
        M<T> target_wide = temp_target_wide.template cast<T>();

        M<double> temp_target_tall ({
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8}
        });
        M<T> target_tall = temp_target_tall.template cast<T>();

        fs::path wide_file = read_matrix_dir / fs::path("wide.csv");
        fs::path tall_file = read_matrix_dir / fs::path("tall.csv");

        M<T> test_wide(read_matrixCSV<M, T>(wide_file));
        M<T> test_tall(read_matrixCSV<M, T>(tall_file));

        ASSERT_MATRIX_NEAR(test_wide, target_wide, Tol<T>::roundoff_T());
        ASSERT_MATRIX_NEAR(test_tall, target_tall, Tol<T>::roundoff_T());

    }

    template <template <typename> typename M>
    void ReadPrecise(
        M<T> target_precise,
        fs::path precise_file
    ) {

        M<T> test_precise(read_matrixCSV<M, T>(precise_file));
        ASSERT_MATRIX_NEAR(test_precise, target_precise, Tol<T>::roundoff_T());

    }

    template <template <typename> typename M, typename T>
    void ReadDifferentThanPrecise(
        M<T> target_precise,
        fs::path precise_file
    ) {

        T eps = static_cast<T>(1.5)*Tol<T>::roundoff_T();
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
    void ReadVector() {

        MatrixVector<T> target({1, 2, 3, 4, 5, 6});

        fs::path vector_file = read_matrix_dir / fs::path("vector.csv");
        MatrixVector<T> test(read_matrixCSV<MatrixVector, T>(vector_file));

        ASSERT_VECTOR_NEAR(test, target, Tol<T>::roundoff_T());

    }

};

TEST_F(MatrixRead_Vector_Test, ReadDoubleVector) { ReadVector<double>(); }
TEST_F(MatrixRead_Vector_Test, ReadSingleVector) { ReadVector<double>(); }
TEST_F(MatrixRead_Vector_Test, ReadHalfVector) { ReadVector<double>(); }

TEST_F(MatrixRead_Vector_Test, FailOnMatrix) {    
    fs::path mat = read_matrix_dir / fs::path("square1.csv");
    try {
        MatrixVector<double> test(read_matrixCSV<MatrixVector, double>(mat));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }
}

// Double type matrix read tests
class MatrixRead_Double_Test: public MatrixRead_T_Test<double> {};

TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>();}
TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(); }
TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrix_Both) {

    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");

    MatrixDense<double> target_precise_dense ({
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    });
    ReadPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<double> target_precise_sparse ({
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    });
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrix_Both) {

    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");

    std::initializer_list<std::initializer_list<double>> li = {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };

    MatrixDense<double> target_precise_dense(li);
    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<double> target_precise_sparse(li);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrixDoubleLimit_Both) {

    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");

    std::initializer_list<std::initializer_list<double>> li = {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };

    MatrixDense<double> target_precise_dense(li);
    ReadPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<double> target_precise_sparse(li);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrixDoubleLimit) {

    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");

    std::initializer_list<std::initializer_list<double>> li = {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };

    MatrixDense<double> target_precise_dense(li);
    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<double> target_precise_sparse(li);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

// Single type matrix read tests
class MatrixRead_Single_Test: public MatrixRead_T_Test<float> {};

TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>();}
TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(); }
TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Single_Test, ReadPreciseMatrix_Both) {

    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");

    std::initializer_list<std::initializer_list<float>> li = {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };

    MatrixDense<float> target_precise_dense(li);
    ReadPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<float> target_precise_sparse(li);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

TEST_F(MatrixRead_Single_Test, ReadDifferentThanPreciseMatrix_Both) {

    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");

    std::initializer_list<std::initializer_list<float>> li = {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };

    MatrixDense<float> target_precise_dense(li);
    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<float> target_precise_sparse(li);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

// Half type matrix read tests
class MatrixRead_Half_Test: public MatrixRead_T_Test<half> {};

TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>();}
TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(); }
TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(); }

TEST_F(MatrixRead_Half_Test, ReadPreciseMatrix) {

    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");

    std::initializer_list<std::initializer_list<half>> li = {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };

    MatrixDense<half> target_precise_dense(li);
    ReadPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<half> target_precise_sparse(li);
    ReadPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}

TEST_F(MatrixRead_Half_Test, ReadDifferentThanPreciseMatrix) {

    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");

    std::initializer_list<std::initializer_list<half>> li = {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };


    MatrixDense<half> target_precise_dense(li);
    ReadDifferentThanPrecise<MatrixDense>(target_precise_dense, precise_file);

    MatrixSparse<half> target_precise_sparse(li);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise_sparse, precise_file);

}
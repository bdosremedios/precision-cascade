#include "test.h"

#include "tools/read_matrix.h"

// General matrix read tests
class read_matrixMTX_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadEmptyMatrix() {

        fs::path empty_file(read_matrix_dir / fs::path("empty_coord.mtx"));
        TMatrix<double> test_empty(read_matrixMTX<TMatrix, double>(
            TestBase::bundle, empty_file
        ));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

        fs::path empty_arr_file(read_matrix_dir / fs::path("empty_array.mtx"));
        TMatrix<double> test_empty_arr(read_matrixMTX<TMatrix, double>(
            TestBase::bundle, empty_arr_file
        ));
        ASSERT_EQ(test_empty_arr.rows(), 0);
        ASSERT_EQ(test_empty_arr.cols(), 0);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadSimpleGeneral() {

        fs::path file_square(read_matrix_dir / fs::path("square_coord.mtx"));
        fs::path file_square_arr(
            read_matrix_dir / fs::path("square_array.mtx")
        );
        TMatrix<TPrecision> target_square(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(4),
              static_cast<TPrecision>(7)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(5),
              static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(6),
              static_cast<TPrecision>(9)}}
        );
        TMatrix<TPrecision> test_square(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_square
        ));
        ASSERT_MATRIX_EQ(test_square, target_square);
        TMatrix<TPrecision> test_square_arr(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_square_arr
        ));
        ASSERT_MATRIX_EQ(test_square_arr, target_square);

        fs::path file_tall(read_matrix_dir / fs::path("tall_coord.mtx"));
        fs::path file_tall_arr(read_matrix_dir / fs::path("tall_array.mtx"));
        TMatrix<TPrecision> target_tall(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(7)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(8)}}
        );
        TMatrix<TPrecision> test_tall(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_tall
        ));
        ASSERT_MATRIX_EQ(test_tall, target_tall);
        TMatrix<TPrecision> test_tall_arr(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_tall_arr
        ));
        ASSERT_MATRIX_EQ(test_tall_arr, target_tall);

        fs::path file_wide(read_matrix_dir / fs::path("wide_coord.mtx"));
        fs::path file_wide_arr(read_matrix_dir / fs::path("wide_array.mtx"));
        TMatrix<TPrecision> target_wide(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(3),
              static_cast<TPrecision>(5), static_cast<TPrecision>(7),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(4),
              static_cast<TPrecision>(6), static_cast<TPrecision>(8),
              static_cast<TPrecision>(10)}}
        );
        TMatrix<TPrecision> test_wide(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_wide
        ));
        ASSERT_MATRIX_EQ(test_wide, target_wide);
        TMatrix<TPrecision> test_wide_arr(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_wide_arr
        ));
        ASSERT_MATRIX_EQ(test_wide_arr, target_wide);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadSimpleSymmetric() {

        fs::path file_symmetric(
            read_matrix_dir / fs::path("square_symmetric.mtx")
        );
        TMatrix<TPrecision> target_symmetric(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(7),
              static_cast<TPrecision>(10), static_cast<TPrecision>(11),
              static_cast<TPrecision>(12)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(8),
              static_cast<TPrecision>(11), static_cast<TPrecision>(13),
              static_cast<TPrecision>(14)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(9),
              static_cast<TPrecision>(12), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)}}
        );
        TMatrix<TPrecision> test_symmetric(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_symmetric
        ));
        ASSERT_MATRIX_EQ(test_symmetric, target_symmetric);

        fs::path file_gen_match(
            read_matrix_dir / fs::path("square_general_match.mtx")
        );
        TMatrix<TPrecision> target_file_to_match(
            read_matrixMTX<TMatrix, TPrecision>(
                TestBase::bundle, file_symmetric
            )
        );
        ASSERT_MATRIX_EQ(test_symmetric, target_file_to_match);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadInteger() {

        fs::path file_square_integer(
            read_matrix_dir / fs::path("square_coord_integer.mtx"));
        fs::path file_square_integer_arr(
            read_matrix_dir / fs::path("square_array_integer.mtx")
        );
        TMatrix<TPrecision> target_square(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(4),
              static_cast<TPrecision>(7)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(5),
              static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(6),
              static_cast<TPrecision>(9)}}
        );
        TMatrix<TPrecision> test_square(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_square_integer
        ));
        ASSERT_MATRIX_EQ(test_square, target_square);
        TMatrix<TPrecision> test_square_arr(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, file_square_integer_arr
        ));
        ASSERT_MATRIX_EQ(test_square_arr, target_square);

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadPreciseGeneral(fs::path mtx_path, TMatrix<TPrecision> target) {

        Scalar<TPrecision> permutation(
            static_cast<TPrecision>(1.5) *
            Tol<TPrecision>::roundoff_T()
        );
        TMatrix<TPrecision> test(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, mtx_path
        ));
        ASSERT_MATRIX_NEAR(test, target, Tol<TPrecision>::roundoff_T());
        ASSERT_MATRIX_LT(
            test,
            TMatrix<TPrecision>(
                MatrixDense<TPrecision>(target) +
                MatrixDense<TPrecision>::Ones(
                    TestBase::bundle, target.rows(), target.cols()
                )*permutation
            )
        );
        ASSERT_MATRIX_GT(
            test,
            TMatrix<TPrecision>(
                MatrixDense<TPrecision>(target) -
                MatrixDense<TPrecision>::Ones(
                    TestBase::bundle, target.rows(), target.cols()
                )*permutation
            )
        );
        
    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void ReadPreciseSymmetric(
        fs::path sym_mtx_path, TMatrix<TPrecision> target
    ) {

        Scalar<TPrecision> permutation(
            static_cast<TPrecision>(1.5) *
            Tol<TPrecision>::roundoff_T()
        );
        TMatrix<TPrecision> test(read_matrixMTX<TMatrix, TPrecision>(
            TestBase::bundle, sym_mtx_path
        ));
        ASSERT_MATRIX_NEAR(test, target, Tol<TPrecision>::roundoff_T());
        ASSERT_MATRIX_LT(
            test,
            TMatrix<TPrecision>(
                MatrixDense<TPrecision>(target) +
                MatrixDense<TPrecision>::Ones(
                    TestBase::bundle, target.rows(), target.cols()
                )*permutation
            )
        );
        ASSERT_MATRIX_GT(
            test,
            TMatrix<TPrecision>(
                MatrixDense<TPrecision>(target) -
                MatrixDense<TPrecision>::Ones(
                    TestBase::bundle, target.rows(), target.cols()
                )*permutation
            )
        );
        
    }

    template <template <typename> typename TMatrix>
    void BadReadMTX() {

        auto missing_file = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("asdasd.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, missing_file);

        auto wrong_extension = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad1")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, wrong_extension);

        auto bad_header_1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header_1);
        auto bad_header2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header2);
        auto bad_header3 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header3.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header3);
        auto bad_header4 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header4.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header4);
        auto bad_header5 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header5.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header5);
        auto bad_header6 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header6.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header6);
        auto bad_header7 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_header7.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_header7);

        auto bad_matrixdim1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim1);
        auto bad_matrixdim2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim2);
        auto bad_matrixdim3 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim3.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim3);
        auto bad_matrixdim4 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim4.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim4);
        auto bad_matrixdim5 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim5.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim5);
        auto bad_matrixdim6 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim6.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim6);
        auto bad_matrixdim7 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim7.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim7);
        auto bad_matrixdim8 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixdim8.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixdim8);

        auto bad_arraydim1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_arraydim1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arraydim1);
        auto bad_arraydim2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_arraydim2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arraydim2);
        auto bad_arraydim3 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_arraydim3.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arraydim3);
        auto bad_arraydim4 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_arraydim4.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arraydim4);

        auto bad_matrixentry1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_matrixentry1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry1);
        auto bad_matrixentry2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_matrixentry2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry2);
        auto bad_matrixentry3 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_matrixentry3.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry3);
        auto bad_matrixentry4 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry4.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry4);
        auto bad_matrixentry5 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry5.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry5);
        auto bad_matrixentry6 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry6.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry6);
        auto bad_matrixentry7 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry7.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry7);
        auto bad_matrixentry8 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry8.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry8);
        auto bad_matrixentry9 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_matrixentry9.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_matrixentry9);

        auto bad_arrayentry1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_arrayentry1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arrayentry1);
        auto bad_arrayentry2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_arrayentry2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arrayentry2);
        auto bad_arrayentry3 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle, read_matrix_dir /
                fs::path("bad_arrayentry3.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_arrayentry3);

        auto bad_wrongorder1 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_wrongorder1.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_wrongorder1);
        auto bad_wrongorder2 = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_wrongorder2.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_wrongorder2);

        auto bad_symmetricinvalidentry = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_symmetricinvalidentry.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_symmetricinvalidentry);

        auto bad_symmetricarray = [this]() {
            read_matrixMTX<TMatrix, double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("bad_symmetric_array.mtx")
            );
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, bad_symmetricarray);

    }

};

TEST_F(read_matrixMTX_Test, ReadEmptyMatrix) {
    ReadEmptyMatrix<MatrixDense, __half>();
    ReadEmptyMatrix<NoFillMatrixSparse, __half>();
    ReadEmptyMatrix<MatrixDense, float>();
    ReadEmptyMatrix<NoFillMatrixSparse, float>();
    ReadEmptyMatrix<MatrixDense, double>();
    ReadEmptyMatrix<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleGeneral) {
    ReadSimpleGeneral<MatrixDense, __half>();
    ReadSimpleGeneral<NoFillMatrixSparse, __half>();
    ReadSimpleGeneral<MatrixDense, float>();
    ReadSimpleGeneral<NoFillMatrixSparse, float>();
    ReadSimpleGeneral<MatrixDense, double>();
    ReadSimpleGeneral<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadSimpleSymmetric) {
    ReadSimpleSymmetric<MatrixDense, __half>();
    ReadSimpleSymmetric<NoFillMatrixSparse, __half>();
    ReadSimpleSymmetric<MatrixDense, float>();
    ReadSimpleSymmetric<NoFillMatrixSparse, float>();
    ReadSimpleSymmetric<MatrixDense, double>();
    ReadSimpleSymmetric<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadInteger) {
    ReadInteger<MatrixDense, __half>();
    ReadInteger<NoFillMatrixSparse, __half>();
    ReadInteger<MatrixDense, float>();
    ReadInteger<NoFillMatrixSparse, float>();
    ReadInteger<MatrixDense, double>();
    ReadInteger<NoFillMatrixSparse, double>();
}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Double) {
    
    MatrixDense<double> mat(
        TestBase::bundle,
        {{static_cast<double>(1.1234567890123450),
          static_cast<double>(1.1234567890123454),
          static_cast<double>(1.1234567890123458),
          static_cast<double>(1.1234567890123452)},
         {static_cast<double>(1.1234567890123451),
          static_cast<double>(1.1234567890123455),
          static_cast<double>(1.1234567890123459),
          static_cast<double>(1.1234567890123453)},
         {static_cast<double>(1.1234567890123452),
          static_cast<double>(1.1234567890123456),
          static_cast<double>(1.1234567890123450),
          static_cast<double>(1.1234567890123454)},
         {static_cast<double>(1.1234567890123453),
          static_cast<double>(1.1234567890123457),
          static_cast<double>(1.1234567890123451),
          static_cast<double>(1.1234567890123455)}}
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise_coord.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise_coord.mtx"),
        NoFillMatrixSparse<double>(mat)
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise_array.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("double_precise_array.mtx"),
        NoFillMatrixSparse<double>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Double) {

    MatrixDense<double> mat(
        TestBase::bundle,
        {{static_cast<double>(1.1234567890123450),
          static_cast<double>(1.1234567890123451),
          static_cast<double>(1.1234567890123452),
          static_cast<double>(1.1234567890123453)},
         {static_cast<double>(1.1234567890123451),
          static_cast<double>(1.1234567890123454),
          static_cast<double>(1.1234567890123455),
          static_cast<double>(1.1234567890123456)},
         {static_cast<double>(1.1234567890123452),
          static_cast<double>(1.1234567890123455),
          static_cast<double>(1.1234567890123457),
          static_cast<double>(1.1234567890123458)},
         {static_cast<double>(1.1234567890123453),
          static_cast<double>(1.1234567890123456),
          static_cast<double>(1.1234567890123458),
          static_cast<double>(1.1234567890123459)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("double_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("double_precise_symmetric.mtx"),
        NoFillMatrixSparse<double>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Single) {

    MatrixDense<float> mat(
        TestBase::bundle,
        {{static_cast<float>(1.12345670), static_cast<float>(1.12345674),
          static_cast<float>(1.12345678), static_cast<float>(1.12345672)},
         {static_cast<float>(1.12345671), static_cast<float>(1.12345675),
          static_cast<float>(1.12345679), static_cast<float>(1.12345673)},
         {static_cast<float>(1.12345672), static_cast<float>(1.12345676),
          static_cast<float>(1.12345670), static_cast<float>(1.12345674)},
         {static_cast<float>(1.12345673), static_cast<float>(1.12345677),
          static_cast<float>(1.12345671), static_cast<float>(1.12345675)}}

    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise_coord.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise_coord.mtx"),
        NoFillMatrixSparse<float>(mat)
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise_array.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("single_precise_array.mtx"),
        NoFillMatrixSparse<float>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Single) {
    
    MatrixDense<float> mat(
        TestBase::bundle,
        {{static_cast<float>(1.12345670), static_cast<float>(1.12345671),
          static_cast<float>(1.12345672), static_cast<float>(1.12345673)},
         {static_cast<float>(1.12345671), static_cast<float>(1.12345674),
          static_cast<float>(1.12345675), static_cast<float>(1.12345676)},
         {static_cast<float>(1.12345672), static_cast<float>(1.12345675),
          static_cast<float>(1.12345677), static_cast<float>(1.12345678)},
         {static_cast<float>(1.12345673), static_cast<float>(1.12345676),
          static_cast<float>(1.12345678), static_cast<float>(1.12345679)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("single_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("single_precise_symmetric.mtx"),
        NoFillMatrixSparse<float>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseGeneral_Half) {

    MatrixDense<__half> mat(
        TestBase::bundle,
        {{static_cast<__half>(1.120), static_cast<__half>(1.124),
          static_cast<__half>(1.128), static_cast<__half>(1.122)},
         {static_cast<__half>(1.121), static_cast<__half>(1.125),
          static_cast<__half>(1.129), static_cast<__half>(1.123)},
         {static_cast<__half>(1.122), static_cast<__half>(1.126),
          static_cast<__half>(1.120), static_cast<__half>(1.124)},
         {static_cast<__half>(1.123), static_cast<__half>(1.127),
          static_cast<__half>(1.121), static_cast<__half>(1.125)}}
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise_coord.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise_coord.mtx"),
        NoFillMatrixSparse<__half>(mat)
    );

    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise_array.mtx"),
        mat
    );
    ReadPreciseGeneral(
        read_matrix_dir / fs::path("half_precise_array.mtx"),
        NoFillMatrixSparse<__half>(mat)
    );

}

TEST_F(read_matrixMTX_Test, ReadPreciseSymmetric_Half) {

    MatrixDense<__half> mat(
        TestBase::bundle,
        {{static_cast<__half>(1.120), static_cast<__half>(1.121),
          static_cast<__half>(1.122), static_cast<__half>(1.123)},
         {static_cast<__half>(1.121), static_cast<__half>(1.124),
          static_cast<__half>(1.125), static_cast<__half>(1.126)},
         {static_cast<__half>(1.122), static_cast<__half>(1.125),
          static_cast<__half>(1.127), static_cast<__half>(1.128)},
         {static_cast<__half>(1.123), static_cast<__half>(1.126),
          static_cast<__half>(1.128), static_cast<__half>(1.129)}}
    );

    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("half_precise_symmetric.mtx"),
        mat
    );
    ReadPreciseSymmetric(
        read_matrix_dir / fs::path("half_precise_symmetric.mtx"),
        NoFillMatrixSparse<__half>(mat)
    );

}

TEST_F(read_matrixMTX_Test, BadReadMTX) {
    BadReadMTX<MatrixDense>();
    BadReadMTX<NoFillMatrixSparse>();
}
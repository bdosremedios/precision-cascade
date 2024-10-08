#include "test.h"

#include "tools/read_matrix.h"

class read_vectorMTX_Test: public TestBase
{
public:

    template <typename TPrecision>
    void ReadFirstSparseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector1.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        Vector<TPrecision> target(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0)}
        );

        ASSERT_VECTOR_NEAR(
            test,
            target,
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadFirstDenseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector2.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        Vector<TPrecision> target(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
             static_cast<TPrecision>(3), static_cast<TPrecision>(4),
             static_cast<TPrecision>(5), static_cast<TPrecision>(6)}
        );

        ASSERT_VECTOR_NEAR(
            test,
            target,
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadRandomSparseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector1.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(2),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(4)}}
        );

        ASSERT_TRUE(
            (test == target_mat.get_col(0).copy_to_vec()) ||
            (test == target_mat.get_col(1).copy_to_vec()) ||
            (test == target_mat.get_col(2).copy_to_vec()) ||
            (test == target_mat.get_col(3).copy_to_vec())
        );

    }

    template <typename TPrecision>
    void ReadRandomDenseVector() {

        fs::path vector_file(read_matrix_dir / fs::path("vector2.mtx"));

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(6),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(5),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(3),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(2),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );

        ASSERT_TRUE(
            (test == target_mat.get_col(0).copy_to_vec()) ||
            (test == target_mat.get_col(1).copy_to_vec()) ||
            (test == target_mat.get_col(2).copy_to_vec())
        );

    }

    template <typename TPrecision>
    void ReadSingleArrayVector() {

        fs::path vector_file(
            read_matrix_dir / fs::path("vector_array_single.mtx")
        );

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        Vector<TPrecision> target(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
             static_cast<TPrecision>(3), static_cast<TPrecision>(4),
             static_cast<TPrecision>(5)}
        );

        ASSERT_VECTOR_NEAR(test, target, Tol<TPrecision>::roundoff_T());

    }

    template <typename TPrecision>
    void ReadFirstArrayVector() {

        fs::path vector_file(
            read_matrix_dir / fs::path("vector_array_mult.mtx")
        );

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "first"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-0.1),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(-0.2),
              static_cast<TPrecision>(20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(-0.3),
              static_cast<TPrecision>(30)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(-0.4),
              static_cast<TPrecision>(40)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(-0.5),
              static_cast<TPrecision>(50)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(-0.6),
              static_cast<TPrecision>(60)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(-0.7),
              static_cast<TPrecision>(70)}}
        );

        ASSERT_VECTOR_NEAR(
            test,
            target_mat.get_col(0).copy_to_vec(),
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadRandomArrayVector() {

        fs::path vector_file(
            read_matrix_dir / fs::path("vector_array_mult.mtx")
        );

        Vector<TPrecision> test(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-0.1),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(-0.2),
              static_cast<TPrecision>(20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(-0.3),
              static_cast<TPrecision>(30)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(-0.4),
              static_cast<TPrecision>(40)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(-0.5),
              static_cast<TPrecision>(50)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(-0.6),
              static_cast<TPrecision>(60)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(-0.7),
              static_cast<TPrecision>(70)}}
        );

        ASSERT_TRUE(
            (test == target_mat.get_col(0).copy_to_vec()) ||
            (test == target_mat.get_col(1).copy_to_vec()) ||
            (test == target_mat.get_col(2).copy_to_vec())
        );

    }

    template <typename TPrecision>
    void ReadFirstIntegerVector() {

        fs::path vector_coord_file(
            read_matrix_dir /
            fs::path("vector_coord_integer.mtx")
        );

        Vector<TPrecision> test_vector_coord(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_coord_file, "first"
        ));

        NoFillMatrixSparse<TPrecision> target_mat_coord(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(2),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(4)}}
        );

        ASSERT_VECTOR_NEAR(
            test_vector_coord,
            target_mat_coord.get_col(0).copy_to_vec(),
            Tol<TPrecision>::roundoff_T()
        );

        fs::path vector_arr_file(
            read_matrix_dir / fs::path("vector_array_integer.mtx")
        );

        Vector<TPrecision> test_vector_arr(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_arr_file, "first"
        ));

        NoFillMatrixSparse<TPrecision> target_mat_arr(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-0.1),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(-0.2),
              static_cast<TPrecision>(20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(-0.3),
              static_cast<TPrecision>(30)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(-0.4),
              static_cast<TPrecision>(40)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(-0.5),
              static_cast<TPrecision>(50)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(-0.6),
              static_cast<TPrecision>(60)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(-0.7),
              static_cast<TPrecision>(70)}}
        );

        ASSERT_VECTOR_NEAR(
            test_vector_arr,
            target_mat_arr.get_col(0).copy_to_vec(),
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void ReadRandomIntegerVector() {

        fs::path vector_coord_file(
            read_matrix_dir /
            fs::path("vector_coord_integer.mtx")
        );

        Vector<TPrecision> test_vector_coord(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_coord_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat_coord(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(2),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(4)}}
        );

        ASSERT_TRUE(
            (test_vector_coord == target_mat_coord.get_col(0).copy_to_vec()) ||
            (test_vector_coord == target_mat_coord.get_col(1).copy_to_vec()) ||
            (test_vector_coord == target_mat_coord.get_col(2).copy_to_vec()) ||
            (test_vector_coord == target_mat_coord.get_col(3).copy_to_vec())
        );

        fs::path vector_arr_file(
            read_matrix_dir / fs::path("vector_array_integer.mtx")
        );

        Vector<TPrecision> test_vector_arr(read_vectorMTX<TPrecision>(
            TestBase::bundle, vector_arr_file, "random"
        ));

        NoFillMatrixSparse<TPrecision> target_mat_arr(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-1),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(-2),
              static_cast<TPrecision>(20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(-3),
              static_cast<TPrecision>(30)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(-4),
              static_cast<TPrecision>(40)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(-5),
              static_cast<TPrecision>(50)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(-6),
              static_cast<TPrecision>(60)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(-7),
              static_cast<TPrecision>(70)}}
        );

        ASSERT_TRUE(
            (test_vector_arr == target_mat_arr.get_col(0).copy_to_vec()) ||
            (test_vector_arr == target_mat_arr.get_col(1).copy_to_vec()) ||
            (test_vector_arr == target_mat_arr.get_col(2).copy_to_vec())
        );

    }

};

TEST_F(read_vectorMTX_Test, ReadFirstSparseVector) {
    ReadFirstSparseVector<double>();
    ReadFirstSparseVector<float>();
    ReadFirstSparseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadFirstDenseVector) {
    ReadFirstDenseVector<double>();
    ReadFirstDenseVector<float>();
    ReadFirstDenseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomSparseVector) {
    ReadRandomSparseVector<double>();
    ReadRandomSparseVector<float>();
    ReadRandomSparseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomDenseVector) {
    ReadRandomDenseVector<double>();
    ReadRandomDenseVector<float>();
    ReadRandomDenseVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadSingleArrayVector) {
    ReadSingleArrayVector<double>();
    ReadSingleArrayVector<float>();
    ReadSingleArrayVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadFirstArrayVector) {
    ReadFirstArrayVector<double>();
    ReadFirstArrayVector<float>();
    ReadFirstArrayVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomArrayVector) {
    ReadRandomArrayVector<double>();
    ReadRandomArrayVector<float>();
    ReadRandomArrayVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadFirstIntegerVector) {
    ReadFirstIntegerVector<double>();
    ReadFirstIntegerVector<float>();
    ReadFirstIntegerVector<__half>();
}

TEST_F(read_vectorMTX_Test, ReadRandomIntegerVector) {
    ReadRandomIntegerVector<double>();
    ReadRandomIntegerVector<float>();
    ReadRandomIntegerVector<__half>();
}

TEST_F(read_vectorMTX_Test, FailOnBadSelection) {

    CHECK_FUNC_HAS_RUNTIME_ERROR(
        print_errors,
        []() {
            read_vectorMTX<double>(
                TestBase::bundle,
                read_matrix_dir / fs::path("vector1.mtx"),
                "waffles"
            );
        }
    );

}
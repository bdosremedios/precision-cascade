#include "../test.h"

#include "types/MatrixVector.h"

class MatrixVector_Test: public TestBase
{
public:

    template <typename T>
    void TestElementAccess() {
        
        constexpr int n(10);
        MatrixVector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_n.get_elem(i), static_cast<T>(i*i)); }
        
        constexpr int m(18);
        MatrixVector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }
        for (int i=0; i<m; ++i) { ASSERT_EQ(test_vec_m.get_elem(i), static_cast<T>(2*i*i-m)); }
        
        constexpr int cnt(10);
        MatrixVector<T> test_vec_cnt(*handle_ptr, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }
        for (int i=0; i<cnt; ++i) { ASSERT_EQ(test_vec_cnt.get_elem(i), static_cast<T>(1+i)); }

    }

    void TestBadElementAccess() {

        const int m(27);
        MatrixVector<double> vec(*handle_ptr, m);
        
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec.get_elem(-1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec.get_elem(m); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec.set_elem(-1, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec.set_elem(m, 0.); });

    }

    template <typename T>
    void TestSlice() {

        constexpr int n(10);
        MatrixVector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        constexpr int m(18);
        MatrixVector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        constexpr int cnt(10);
        MatrixVector<T> test_vec_cnt(*handle_ptr, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }

        MatrixVector<T> test_vec_cnt_2_6(test_vec_cnt.slice(2, 6));
        ASSERT_EQ(test_vec_cnt_2_6.rows(), 6);
        for (int i=0; i<6; ++i) { ASSERT_EQ(test_vec_cnt_2_6.get_elem(i),
                                            test_vec_cnt.get_elem(i+2)); }
        
        MatrixVector<T> test_vec_m_1_3(test_vec_m.slice(1, 3));
        ASSERT_EQ(test_vec_m_1_3.rows(), 3);
        for (int i=0; i<3; ++i) { ASSERT_EQ(test_vec_m_1_3.get_elem(i),
                                            test_vec_m.get_elem(i+1)); }

        MatrixVector<T> test_vec_n_0_4(test_vec_n.slice(0, 4));
        ASSERT_EQ(test_vec_n_0_4.rows(), 4);
        for (int i=0; i<4; ++i) { ASSERT_EQ(test_vec_n_0_4.get_elem(i),
                                            test_vec_n.get_elem(i)); }

        MatrixVector<T> test_vec_m_dupe(test_vec_m.slice(0, m));
        ASSERT_EQ(test_vec_m_dupe.rows(), m);
        ASSERT_VECTOR_EQ(test_vec_m_dupe, test_vec_m);

        MatrixVector<T> test_vec_slice_empty(test_vec_m.slice(0, 0));
        ASSERT_EQ(test_vec_slice_empty.rows(), 0);

        MatrixVector<T> test_vec_slice_empty_2(test_vec_n.slice(1, 0));
        ASSERT_EQ(test_vec_slice_empty_2.rows(), 0);

    }

    template <typename T>
    void TestBadSlice() {

        constexpr int m(18);
        MatrixVector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { test_vec_m.slice(1, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { test_vec_m.slice(-1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { test_vec_m.slice(m, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { test_vec_m.slice(0, m+1); });
        
    }

    template <typename T>
    void TestPropertyMethods() {
        
        constexpr int m(8);
        MatrixVector<T> test_vec_m(*handle_ptr, m);

        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);
        
        constexpr int n(16);
        MatrixVector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        ASSERT_EQ(test_vec_n.rows(), n);
        ASSERT_EQ(test_vec_n.cols(), 1);

    } 

    void TestConstruction() {

        MatrixVector<double> test_vec_0(*handle_ptr, 0);
        ASSERT_EQ(test_vec_0.rows(), 0);
        ASSERT_EQ(test_vec_0.cols(), 1);

        constexpr int m(24);
        MatrixVector<double> test_vec_m(*handle_ptr, m);
        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);

    }

    template <typename T>
    void TestListInitialization() {
        
        MatrixVector<T> test_vec_6(
            *handle_ptr,
            {static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.),
             static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)}
        );
        ASSERT_EQ(test_vec_6.get_elem(0), static_cast<T>(7.));
        ASSERT_EQ(test_vec_6.get_elem(1), static_cast<T>(5.));
        ASSERT_EQ(test_vec_6.get_elem(2), static_cast<T>(3.));
        ASSERT_EQ(test_vec_6.get_elem(3), static_cast<T>(1.));
        ASSERT_EQ(test_vec_6.get_elem(4), static_cast<T>(6.));
        ASSERT_EQ(test_vec_6.get_elem(5), static_cast<T>(2.));

        MatrixVector<T> test_vec_empty(*handle_ptr, {});

    }

    template <typename T>
    void TestCopyAssignment() {

        MatrixVector<T> test_vec_empty(*handle_ptr, {});
        MatrixVector<T> test_vec_4(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(10.)}
        );
        MatrixVector<T> test_vec_5(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.),
             static_cast<T>(10.), static_cast<T>(0.)}
        );
        MatrixVector<T> test_vec_6(
            *handle_ptr,
            {static_cast<T>(12.), static_cast<T>(12.), static_cast<T>(14.),
             static_cast<T>(14.), static_cast<T>(12.), static_cast<T>(12.)}
        );

        // Copy to empty
        test_vec_empty = test_vec_6;
        ASSERT_EQ(test_vec_empty.rows(), test_vec_6.rows());
        ASSERT_VECTOR_EQ(test_vec_empty, test_vec_6);

        // Copy to populated
        test_vec_4 = test_vec_6;
        ASSERT_EQ(test_vec_4.rows(), test_vec_6.rows());
        ASSERT_VECTOR_EQ(test_vec_4, test_vec_6);

        // Reassignment
        test_vec_4 = test_vec_5;
        ASSERT_EQ(test_vec_4.rows(), test_vec_5.rows());
        ASSERT_VECTOR_EQ(test_vec_4, test_vec_5);

        // Transitive assignment
        test_vec_empty = test_vec_4;
        ASSERT_EQ(test_vec_empty.rows(), test_vec_5.rows());
        ASSERT_VECTOR_EQ(test_vec_empty, test_vec_5);

        // Self-assignment
        test_vec_6 = test_vec_6;
        ASSERT_EQ(test_vec_6.rows(), 6);
        ASSERT_EQ(test_vec_6.get_elem(0), static_cast<T>(12.));
        ASSERT_EQ(test_vec_6.get_elem(1), static_cast<T>(12.));
        ASSERT_EQ(test_vec_6.get_elem(2), static_cast<T>(14.));
        ASSERT_EQ(test_vec_6.get_elem(3), static_cast<T>(14.));
        ASSERT_EQ(test_vec_6.get_elem(4), static_cast<T>(12.));
        ASSERT_EQ(test_vec_6.get_elem(5), static_cast<T>(12.));

    }

    template <typename T>
    void TestCopyConstruction() {

        MatrixVector<T> test_vec_4(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(10.)}
        );

        MatrixVector<T> test_vec_copied(test_vec_4);
        ASSERT_EQ(test_vec_copied.rows(), 4);
        ASSERT_EQ(test_vec_copied.get_elem(0), static_cast<T>(-3.));
        ASSERT_EQ(test_vec_copied.get_elem(1), static_cast<T>(0.));
        ASSERT_EQ(test_vec_copied.get_elem(2), static_cast<T>(1.));
        ASSERT_EQ(test_vec_copied.get_elem(3), static_cast<T>(10.));

    }

    template <typename T>
    void TestStaticCreation() {

        constexpr int m_zero(15);
        MatrixVector<T> test_zero(MatrixVector<T>::Zero(*handle_ptr, m_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        for (int i=0; i<m_zero; ++i) { ASSERT_EQ(test_zero.get_elem(i), static_cast<T>(0.)); }

        constexpr int m_one(15);
        MatrixVector<T> test_ones(MatrixVector<T>::Ones(*handle_ptr, m_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        for (int i=0; i<m_one; ++i) { ASSERT_EQ(test_ones.get_elem(i), static_cast<T>(1.)); }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from other 5 (so only will fail if some 5 numbers in a row are exactly
        // the same))
        constexpr int m_rand(200);
        MatrixVector<T> test_rand(MatrixVector<T>::Random(*handle_ptr, m_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        for (int i=2; i<m_one-2; ++i) {
            ASSERT_TRUE(
                ((test_rand.get_elem(i) != test_rand.get_elem(i-2)) ||
                 (test_rand.get_elem(i) != test_rand.get_elem(i-1)) ||
                 (test_rand.get_elem(i) != test_rand.get_elem(i+1)) ||
                 (test_rand.get_elem(i) != test_rand.get_elem(i+2)))
            );
        }

    }

    template <typename T>
    void TestScale() {

        MatrixVector<T> vec(
            *handle_ptr,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        MatrixVector<T> vec_scaled_mult = vec*static_cast<T>(4);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec_scaled_mult.get_elem(i),
                                            static_cast<T>(4)*vec.get_elem(i)); }

        MatrixVector<T> vec_scaled_div = vec/static_cast<T>(10);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec_scaled_div.get_elem(i),
                                            (static_cast<T>(1)/static_cast<T>(10))*vec.get_elem(i)); }

    }

    template <typename T>
    void TestScaleAssignment() {

        MatrixVector<T> orig_vec(
            *handle_ptr,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        MatrixVector<T> vec(orig_vec);
        vec *= static_cast<T>(3);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec.get_elem(i),
                                            static_cast<T>(3)*orig_vec.get_elem(i)); }

        vec = orig_vec;
        vec /= static_cast<T>(5);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec.get_elem(i),
                                            (static_cast<T>(1)/static_cast<T>(5))*orig_vec.get_elem(i)); }

    }

    template <typename T>
    void TestAddSub() {

        MatrixVector<T> vec_1(
            *handle_ptr,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        MatrixVector<T> vec_2(
            *handle_ptr,
            {static_cast<T>(-38.), static_cast<T>(0.5), static_cast<T>(-0.6)}
        );

        MatrixVector<T> vec_add = vec_1+vec_2;
        ASSERT_EQ(vec_add.get_elem(0), static_cast<T>(-80.));
        ASSERT_EQ(vec_add.get_elem(1), static_cast<T>(0.5));
        ASSERT_EQ(vec_add.get_elem(2), static_cast<T>(0.));

        MatrixVector<T> vec_sub_1 = vec_1-vec_2;
        ASSERT_EQ(vec_sub_1.get_elem(0), static_cast<T>(-4.));
        ASSERT_EQ(vec_sub_1.get_elem(1), static_cast<T>(-0.5));
        ASSERT_EQ(vec_sub_1.get_elem(2), static_cast<T>(1.2));

        MatrixVector<T> vec_sub_2 = vec_2-vec_1;
        ASSERT_EQ(vec_sub_2.get_elem(0), static_cast<T>(4.));
        ASSERT_EQ(vec_sub_2.get_elem(1), static_cast<T>(0.5));
        ASSERT_EQ(vec_sub_2.get_elem(2), static_cast<T>(-1.2));

    }

    template <typename T>
    void TestAddSubAssignment() {

        MatrixVector<T> vec_1(
            *handle_ptr,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        MatrixVector<T> vec_2(
            *handle_ptr,
            {static_cast<T>(-38.), static_cast<T>(0.5), static_cast<T>(-0.6)}
        );

        MatrixVector<T> vec_3(vec_1);
        vec_3 += vec_2;
        ASSERT_VECTOR_EQ(vec_3, vec_1+vec_2);
        vec_3 += vec_3;
        ASSERT_VECTOR_EQ(vec_3, (vec_1+vec_2)*static_cast<T>(2.));

        MatrixVector<T> vec_4(vec_1);
        vec_4 -= vec_2;
        ASSERT_VECTOR_EQ(vec_4, vec_1-vec_2);
        vec_4 -= vec_4;
        ASSERT_VECTOR_EQ(vec_4, (vec_1+vec_2)*static_cast<T>(0.));

    }

    template <typename T>
    void TestDot() {

        // Pre-calculated
        MatrixVector<T> vec_1_dot(
            *handle_ptr,
            {static_cast<T>(-4.), static_cast<T>(3.4), static_cast<T>(0.),
             static_cast<T>(-2.1), static_cast<T>(1.8)}
        );
        MatrixVector<T> vec_2_dot(
            *handle_ptr,
            {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(1.5),
             static_cast<T>(-4.5), static_cast<T>(2.)}
        );
        ASSERT_NEAR(static_cast<double>(vec_1_dot.dot(vec_2_dot)),
                    11.05,
                    11.05*Tol<T>::gamma(5));

        // Random
        MatrixVector<T> vec_1_dot_r(MatrixVector<T>::Random(*handle_ptr, 10));
        MatrixVector<T> vec_2_dot_r(MatrixVector<T>::Random(*handle_ptr, 10));
        T acc = static_cast<T>(0.);
        for (int i=0; i<10; ++i) { acc += vec_1_dot_r.get_elem(i)*vec_2_dot_r.get_elem(i); }
        ASSERT_NEAR(static_cast<double>(vec_1_dot_r.dot(vec_2_dot_r)),
                    static_cast<double>(acc),
                    2.*std::abs(10.*Tol<T>::gamma(10)));

    }

    template <typename T>
    void TestNorm() {

        // Pre-calculated
        MatrixVector<T> vec_norm(
            *handle_ptr, 
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );
        ASSERT_NEAR(vec_norm.norm(),
                    static_cast<T>(9.),
                    static_cast<T>(9.)*static_cast<T>(Tol<T>::gamma(5)));

        // Random
        MatrixVector<T> vec_norm_r(MatrixVector<T>::Random(*handle_ptr, 10));
        ASSERT_NEAR(vec_norm_r.norm(),
                    std::sqrt(vec_norm_r.dot(vec_norm_r)),
                    std::sqrt(vec_norm_r.dot(vec_norm_r))*Tol<T>::gamma(10));

    }

    template <typename T>
    void TestBadVecVecOps() {

        const int m(7);
        MatrixVector<T> vec(MatrixVector<T>::Random(*handle_ptr, m));
        MatrixVector<T> vec_too_small(MatrixVector<T>::Random(*handle_ptr, m-3));
        MatrixVector<T> vec_too_large(MatrixVector<T>::Random(*handle_ptr, m+2));

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec + vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec + vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec - vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec - vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec += vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec += vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec -= vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() mutable { vec -= vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec.dot(vec_too_small); });
        CHECK_FUNC_HAS_RUNTIME_ERROR([=]() { vec.dot(vec_too_large); });

    }

    void TestCast() {
        
        constexpr int m(20);
        MatrixVector<double> vec_dbl(MatrixVector<double>::Random(*handle_ptr, m));

        MatrixVector<float> vec_sgl(vec_dbl.cast<float>());
        ASSERT_EQ(vec_sgl.rows(), m);
        for (int i=0; i<m; ++i) { ASSERT_EQ(vec_sgl.get_elem(i),
                                            static_cast<float>(vec_dbl.get_elem(i))); }

        MatrixVector<__half> vec_hlf(vec_dbl.cast<__half>());
        ASSERT_EQ(vec_hlf.rows(), m);
        for (int i=0; i<m; ++i) { ASSERT_EQ(vec_hlf.get_elem(i),
                                            static_cast<__half>(vec_dbl.get_elem(i))); }

    }

    template <typename T>
    void TestBooleanEqual() {

        MatrixVector<T> vec_to_compare(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        MatrixVector<T> vec_same(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        MatrixVector<T> vec_diffbeg(
            *handle_ptr,
            {static_cast<T>(1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        MatrixVector<T> vec_diffend(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(70)}
        );
        MatrixVector<T> vec_diffmid(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(-100.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        MatrixVector<T> vec_smaller(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101)}
        );
        MatrixVector<T> vec_bigger(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(0)}
        );
        MatrixVector<T> vec_empty_1(*handle_ptr, {});
        MatrixVector<T> vec_empty_2(*handle_ptr, {});

        ASSERT_TRUE(vec_to_compare == vec_to_compare);
        ASSERT_TRUE(vec_to_compare == vec_same);
        ASSERT_FALSE(vec_to_compare == vec_diffbeg);
        ASSERT_FALSE(vec_to_compare == vec_diffend);
        ASSERT_FALSE(vec_to_compare == vec_diffmid);
        ASSERT_FALSE(vec_to_compare == vec_smaller);
        ASSERT_FALSE(vec_to_compare == vec_bigger);
        ASSERT_TRUE(vec_empty_1 == vec_empty_2);

    }

};

TEST_F(MatrixVector_Test, TestElementAccess) {
    TestElementAccess<__half>();
    TestElementAccess<float>();
    TestElementAccess<double>();
}

TEST_F(MatrixVector_Test, TestBadElementAccess) { TestBadElementAccess(); }

TEST_F(MatrixVector_Test, TestSlice) {
    TestSlice<__half>(); TestSlice<float>(); TestSlice<double>();
}

TEST_F(MatrixVector_Test, TestBadSlice) {
    TestBadSlice<__half>(); TestBadSlice<float>(); TestBadSlice<double>();
}

TEST_F(MatrixVector_Test, TestPropertyMethods) {
    TestPropertyMethods<__half>();
    TestPropertyMethods<float>();
    TestPropertyMethods<double>();
}

TEST_F(MatrixVector_Test, TestConstruction) { TestConstruction(); }

TEST_F(MatrixVector_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixVector_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(MatrixVector_Test, TestCopyConstruction) {
    TestCopyConstruction<__half>();
    TestCopyConstruction<float>();
    TestCopyConstruction<double>();
}


TEST_F(MatrixVector_Test, TestStaticCreation) {
    TestStaticCreation<__half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixVector_Test, TestScale) {
    TestScale<__half>(); TestScale<float>(); TestScale<double>();
}
TEST_F(MatrixVector_Test, TestScaleAssignment) {
    TestScaleAssignment<__half>(); TestScaleAssignment<float>(); TestScaleAssignment<double>();
}

TEST_F(MatrixVector_Test, TestAddSub) {
    TestAddSub<__half>(); TestAddSub<float>(); TestAddSub<double>();
}
TEST_F(MatrixVector_Test, TestAddSubAssignment) {
    TestAddSubAssignment<__half>(); TestAddSubAssignment<float>(); TestAddSubAssignment<double>();
}

TEST_F(MatrixVector_Test, TestDot) { TestDot<__half>(); TestDot<float>(); TestDot<double>(); }

TEST_F(MatrixVector_Test, TestNorm) { TestNorm<__half>(); TestNorm<float>(); TestNorm<double>(); }

TEST_F(MatrixVector_Test, TestCast) { TestCast(); }

TEST_F(MatrixVector_Test, TestBadVecVecOps) {
    TestBadVecVecOps<__half>();
    TestBadVecVecOps<float>();
    TestBadVecVecOps<double>();
}

TEST_F(MatrixVector_Test, TestBooleanEqual) {
    TestBooleanEqual<__half>();
    TestBooleanEqual<float>();
    TestBooleanEqual<double>();
}
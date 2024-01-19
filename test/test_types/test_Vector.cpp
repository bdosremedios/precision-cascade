#include "../test.h"

#include "types/Vector.h"

class Vector_Test: public TestBase
{
public:

    template <typename T>
    void TestElementAccess() {
        
        constexpr int n(10);
        Vector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_n.get_elem(i), static_cast<T>(i*i)); }
        
        constexpr int m(18);
        Vector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }
        for (int i=0; i<m; ++i) { ASSERT_EQ(test_vec_m.get_elem(i), static_cast<T>(2*i*i-m)); }
        
        constexpr int cnt(10);
        Vector<T> test_vec_cnt(*handle_ptr, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }
        for (int i=0; i<cnt; ++i) { ASSERT_EQ(test_vec_cnt.get_elem(i), static_cast<T>(1+i)); }

    }

    void TestBadElementAccess() {

        const int m(27);
        Vector<double> vec(*handle_ptr, m);
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.get_elem(-1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.get_elem(m); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec.set_elem(-1, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec.set_elem(m, 0.); });

    }

    template <typename T>
    void TestSlice() {

        constexpr int n(10);
        Vector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        constexpr int m(18);
        Vector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        constexpr int cnt(10);
        Vector<T> test_vec_cnt(*handle_ptr, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }

        Vector<T> test_vec_cnt_2_6(test_vec_cnt.slice(2, 6));
        ASSERT_EQ(test_vec_cnt_2_6.rows(), 6);
        for (int i=0; i<6; ++i) { ASSERT_EQ(test_vec_cnt_2_6.get_elem(i),
                                            test_vec_cnt.get_elem(i+2)); }
        
        Vector<T> test_vec_m_1_3(test_vec_m.slice(1, 3));
        ASSERT_EQ(test_vec_m_1_3.rows(), 3);
        for (int i=0; i<3; ++i) { ASSERT_EQ(test_vec_m_1_3.get_elem(i),
                                            test_vec_m.get_elem(i+1)); }

        Vector<T> test_vec_n_0_4(test_vec_n.slice(0, 4));
        ASSERT_EQ(test_vec_n_0_4.rows(), 4);
        for (int i=0; i<4; ++i) { ASSERT_EQ(test_vec_n_0_4.get_elem(i),
                                            test_vec_n.get_elem(i)); }

        Vector<T> test_vec_m_dupe(test_vec_m.slice(0, m));
        ASSERT_EQ(test_vec_m_dupe.rows(), m);
        ASSERT_VECTOR_EQ(test_vec_m_dupe, test_vec_m);

        Vector<T> test_vec_slice_empty(test_vec_m.slice(0, 0));
        ASSERT_EQ(test_vec_slice_empty.rows(), 0);

        Vector<T> test_vec_slice_empty_2(test_vec_n.slice(1, 0));
        ASSERT_EQ(test_vec_slice_empty_2.rows(), 0);

    }

    template <typename T>
    void TestBadSlice() {

        constexpr int m(18);
        Vector<T> test_vec_m(*handle_ptr, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.slice(1, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.slice(-1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.slice(m, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.slice(0, m+1); });
        
    }

    template <typename T>
    void TestPropertyMethods() {
        
        constexpr int m(8);
        Vector<T> test_vec_m(*handle_ptr, m);

        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);
        
        constexpr int n(16);
        Vector<T> test_vec_n(*handle_ptr, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        ASSERT_EQ(test_vec_n.rows(), n);
        ASSERT_EQ(test_vec_n.cols(), 1);

    } 

    void TestConstruction() {

        Vector<double> test_vec_0(*handle_ptr, 0);
        ASSERT_EQ(test_vec_0.rows(), 0);
        ASSERT_EQ(test_vec_0.cols(), 1);

        constexpr int m(24);
        Vector<double> test_vec_m(*handle_ptr, m);
        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);

    }

    template <typename T>
    void TestListInitialization() {
        
        Vector<T> test_vec_6(
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

        Vector<T> test_vec_empty(*handle_ptr, {});

    }

    template <typename T>
    void TestDynamicMemConstruction() {
    
        const int m_manual(3);
        T *h_vec_manual = static_cast<T *>(malloc(m_manual*sizeof(T)));
        h_vec_manual[0] = static_cast<T>(-5);
        h_vec_manual[1] = static_cast<T>(100);
        h_vec_manual[2] = static_cast<T>(-20);

        Vector<T> test_vec_manual(*handle_ptr, h_vec_manual, m_manual);

        Vector<T> target_vec_manual(
            *handle_ptr,
            {static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)}
        );

        ASSERT_MATRIX_EQ(test_vec_manual, target_vec_manual);

        free(h_vec_manual);
    
        const int m_rand(7);
        T *h_vec_rand = static_cast<T *>(malloc(m_rand*sizeof(T)));
        for (int i=0; i<m_rand; ++i) { h_vec_rand[i] = rand(); }

        Vector<T> test_vec_rand(*handle_ptr, h_vec_rand, m_rand);

        ASSERT_EQ(test_vec_rand.rows(), m_rand);
        ASSERT_EQ(test_vec_rand.cols(), 1);
        for (int i=0; i<m_rand; ++i) {
            ASSERT_EQ(test_vec_rand.get_elem(i), h_vec_rand[i]);
        }

        free(h_vec_rand);

    }

    template <typename T>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(3);

        Vector<T> vec_manual(
            *handle_ptr,
            {static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)}
        );

        T *h_vec_manual = static_cast<T *>(malloc(m_manual*sizeof(T)));
        vec_manual.copy_data_to_ptr(h_vec_manual, m_manual);

        ASSERT_EQ(h_vec_manual[0], static_cast<T>(-5));
        ASSERT_EQ(h_vec_manual[1], static_cast<T>(100));
        ASSERT_EQ(h_vec_manual[2], static_cast<T>(-20));

        free(h_vec_manual);
    
        const int m_rand(7);

        Vector<T> vec_rand(
            *handle_ptr,
            {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
             static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
             static_cast<T>(rand())}
        );

        T *h_vec_rand = static_cast<T *>(malloc(m_rand*sizeof(T)));
        vec_rand.copy_data_to_ptr(h_vec_rand, m_rand);

        for (int i=0; i<m_rand; ++i) { ASSERT_EQ(h_vec_rand[i], vec_rand.get_elem(i)); }

        free(h_vec_rand);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_rand(10);
        Vector<double> vec_rand(*handle_ptr, m_rand);
        double *h_vec_rand = static_cast<double *>(malloc(m_rand*sizeof(double)));
        
        auto try_row_too_small = [=]() { vec_rand.copy_data_to_ptr(h_vec_rand, m_rand-2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_small);

        auto try_row_too_large = [=]() { vec_rand.copy_data_to_ptr(h_vec_rand, m_rand+2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_large);

    }

    template <typename T>
    void TestCopyAssignment() {

        Vector<T> test_vec_empty(*handle_ptr, {});
        Vector<T> test_vec_4(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(10.)}
        );
        Vector<T> test_vec_5(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.),
             static_cast<T>(10.), static_cast<T>(0.)}
        );
        Vector<T> test_vec_6(
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

        Vector<T> test_vec_4(
            *handle_ptr,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(10.)}
        );

        Vector<T> test_vec_copied(test_vec_4);
        ASSERT_EQ(test_vec_copied.rows(), 4);
        ASSERT_EQ(test_vec_copied.get_elem(0), static_cast<T>(-3.));
        ASSERT_EQ(test_vec_copied.get_elem(1), static_cast<T>(0.));
        ASSERT_EQ(test_vec_copied.get_elem(2), static_cast<T>(1.));
        ASSERT_EQ(test_vec_copied.get_elem(3), static_cast<T>(10.));

    }

    template <typename T>
    void TestStaticCreation() {

        constexpr int m_zero(15);
        Vector<T> test_zero(Vector<T>::Zero(*handle_ptr, m_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        for (int i=0; i<m_zero; ++i) { ASSERT_EQ(test_zero.get_elem(i), static_cast<T>(0.)); }

        constexpr int m_one(15);
        Vector<T> test_ones(Vector<T>::Ones(*handle_ptr, m_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        for (int i=0; i<m_one; ++i) { ASSERT_EQ(test_ones.get_elem(i), static_cast<T>(1.)); }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from other 5 (so only will fail if some 5 numbers in a row are exactly
        // the same))
        constexpr int m_rand(200);
        Vector<T> test_rand(Vector<T>::Random(*handle_ptr, m_rand));
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

        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        Vector<T> vec_scaled_mult = vec*static_cast<T>(4);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec_scaled_mult.get_elem(i),
                                            static_cast<T>(4)*vec.get_elem(i)); }

        Vector<T> vec_scaled_div = vec/static_cast<T>(10);
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec_scaled_div.get_elem(i),
                                            (static_cast<T>(1)/static_cast<T>(10))*vec.get_elem(i)); }

    }

    template <typename T>
    void TestScaleAssignment() {

        Vector<T> orig_vec(
            *handle_ptr,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        Vector<T> vec(orig_vec);
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

        Vector<T> vec_1(
            *handle_ptr,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        Vector<T> vec_2(
            *handle_ptr,
            {static_cast<T>(-38.), static_cast<T>(0.5), static_cast<T>(-0.6)}
        );

        Vector<T> vec_add = vec_1+vec_2;
        ASSERT_EQ(vec_add.get_elem(0), static_cast<T>(-80.));
        ASSERT_EQ(vec_add.get_elem(1), static_cast<T>(0.5));
        ASSERT_EQ(vec_add.get_elem(2), static_cast<T>(0.));

        Vector<T> vec_sub_1 = vec_1-vec_2;
        ASSERT_EQ(vec_sub_1.get_elem(0), static_cast<T>(-4.));
        ASSERT_EQ(vec_sub_1.get_elem(1), static_cast<T>(-0.5));
        ASSERT_EQ(vec_sub_1.get_elem(2), static_cast<T>(1.2));

        Vector<T> vec_sub_2 = vec_2-vec_1;
        ASSERT_EQ(vec_sub_2.get_elem(0), static_cast<T>(4.));
        ASSERT_EQ(vec_sub_2.get_elem(1), static_cast<T>(0.5));
        ASSERT_EQ(vec_sub_2.get_elem(2), static_cast<T>(-1.2));

    }

    template <typename T>
    void TestAddSubAssignment() {

        Vector<T> vec_1(
            *handle_ptr,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        Vector<T> vec_2(
            *handle_ptr,
            {static_cast<T>(-38.), static_cast<T>(0.5), static_cast<T>(-0.6)}
        );

        Vector<T> vec_3(vec_1);
        vec_3 += vec_2;
        ASSERT_VECTOR_EQ(vec_3, vec_1+vec_2);
        vec_3 += vec_3;
        ASSERT_VECTOR_EQ(vec_3, (vec_1+vec_2)*static_cast<T>(2.));

        Vector<T> vec_4(vec_1);
        vec_4 -= vec_2;
        ASSERT_VECTOR_EQ(vec_4, vec_1-vec_2);
        vec_4 -= vec_4;
        ASSERT_VECTOR_EQ(vec_4, (vec_1+vec_2)*static_cast<T>(0.));

    }

    template <typename T>
    void TestDot() {

        // Pre-calculated
        Vector<T> vec_1_dot(
            *handle_ptr,
            {static_cast<T>(-4.), static_cast<T>(3.4), static_cast<T>(0.),
             static_cast<T>(-2.1), static_cast<T>(1.8)}
        );
        Vector<T> vec_2_dot(
            *handle_ptr,
            {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(1.5),
             static_cast<T>(-4.5), static_cast<T>(2.)}
        );
        ASSERT_NEAR(static_cast<double>(vec_1_dot.dot(vec_2_dot)),
                    11.05,
                    11.05*Tol<T>::gamma(5));

        // Random
        Vector<T> vec_1_dot_r(Vector<T>::Random(*handle_ptr, 10));
        Vector<T> vec_2_dot_r(Vector<T>::Random(*handle_ptr, 10));
        T acc = static_cast<T>(0.);
        for (int i=0; i<10; ++i) { acc += vec_1_dot_r.get_elem(i)*vec_2_dot_r.get_elem(i); }
        ASSERT_NEAR(static_cast<double>(vec_1_dot_r.dot(vec_2_dot_r)),
                    static_cast<double>(acc),
                    2.*std::abs(10.*Tol<T>::gamma(10)));

    }

    template <typename T>
    void TestNorm() {

        // Pre-calculated
        Vector<T> vec_norm(
            *handle_ptr, 
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );
        ASSERT_NEAR(vec_norm.norm(),
                    static_cast<T>(9.),
                    static_cast<T>(9.)*static_cast<T>(Tol<T>::gamma(5)));

        // Random
        Vector<T> vec_norm_r(Vector<T>::Random(*handle_ptr, 10));
        ASSERT_NEAR(vec_norm_r.norm(),
                    std::sqrt(vec_norm_r.dot(vec_norm_r)),
                    std::sqrt(vec_norm_r.dot(vec_norm_r))*Tol<T>::gamma(10));

    }

    template <typename T>
    void TestBadVecVecOps() {

        const int m(7);
        Vector<T> vec(Vector<T>::Random(*handle_ptr, m));
        Vector<T> vec_too_small(Vector<T>::Random(*handle_ptr, m-3));
        Vector<T> vec_too_large(Vector<T>::Random(*handle_ptr, m+2));

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec + vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec + vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec - vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec - vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec += vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec += vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec -= vec_too_small; });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec -= vec_too_large; });

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.dot(vec_too_small); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.dot(vec_too_large); });

    }

    void TestCast() {
        
        constexpr int m(20);
        Vector<double> vec_dbl(Vector<double>::Random(*handle_ptr, m));

        Vector<float> vec_sgl(vec_dbl.cast<float>());
        ASSERT_EQ(vec_sgl.rows(), m);
        for (int i=0; i<m; ++i) { ASSERT_EQ(vec_sgl.get_elem(i),
                                            static_cast<float>(vec_dbl.get_elem(i))); }

        Vector<__half> vec_hlf(vec_dbl.cast<__half>());
        ASSERT_EQ(vec_hlf.rows(), m);
        for (int i=0; i<m; ++i) { ASSERT_EQ(vec_hlf.get_elem(i),
                                            static_cast<__half>(vec_dbl.get_elem(i))); }

    }

    template <typename T>
    void TestBooleanEqual() {

        Vector<T> vec_to_compare(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_same(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_diffbeg(
            *handle_ptr,
            {static_cast<T>(1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_diffend(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(70)}
        );
        Vector<T> vec_diffmid(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(-100.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_smaller(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101)}
        );
        Vector<T> vec_bigger(
            *handle_ptr,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(0)}
        );
        Vector<T> vec_empty_1(*handle_ptr, {});
        Vector<T> vec_empty_2(*handle_ptr, {});

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

TEST_F(Vector_Test, TestElementAccess) {
    TestElementAccess<__half>();
    TestElementAccess<float>();
    TestElementAccess<double>();
}

TEST_F(Vector_Test, TestBadElementAccess) { TestBadElementAccess(); }

TEST_F(Vector_Test, TestSlice) {
    TestSlice<__half>(); TestSlice<float>(); TestSlice<double>();
}

TEST_F(Vector_Test, TestBadSlice) {
    TestBadSlice<__half>(); TestBadSlice<float>(); TestBadSlice<double>();
}

TEST_F(Vector_Test, TestPropertyMethods) {
    TestPropertyMethods<__half>();
    TestPropertyMethods<float>();
    TestPropertyMethods<double>();
}

TEST_F(Vector_Test, TestConstruction) { TestConstruction(); }

TEST_F(Vector_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(Vector_Test, TestDynamicMemConstruction) {
    TestDynamicMemConstruction<__half>();
    TestDynamicMemConstruction<float>();
    TestDynamicMemConstruction<double>();
}

TEST_F(Vector_Test, TestDynamicMemCopyToPtr) {
    TestDynamicMemCopyToPtr<__half>();
    TestDynamicMemCopyToPtr<float>();
    TestDynamicMemCopyToPtr<double>();
}

TEST_F(Vector_Test, TestBadDynamicMemCopyToPtr) { TestBadDynamicMemCopyToPtr(); }

TEST_F(Vector_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(Vector_Test, TestCopyConstruction) {
    TestCopyConstruction<__half>();
    TestCopyConstruction<float>();
    TestCopyConstruction<double>();
}


TEST_F(Vector_Test, TestStaticCreation) {
    TestStaticCreation<__half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(Vector_Test, TestScale) {
    TestScale<__half>(); TestScale<float>(); TestScale<double>();
}
TEST_F(Vector_Test, TestScaleAssignment) {
    TestScaleAssignment<__half>(); TestScaleAssignment<float>(); TestScaleAssignment<double>();
}

TEST_F(Vector_Test, TestAddSub) {
    TestAddSub<__half>(); TestAddSub<float>(); TestAddSub<double>();
}
TEST_F(Vector_Test, TestAddSubAssignment) {
    TestAddSubAssignment<__half>(); TestAddSubAssignment<float>(); TestAddSubAssignment<double>();
}

TEST_F(Vector_Test, TestDot) { TestDot<__half>(); TestDot<float>(); TestDot<double>(); }

TEST_F(Vector_Test, TestNorm) { TestNorm<__half>(); TestNorm<float>(); TestNorm<double>(); }

TEST_F(Vector_Test, TestCast) { TestCast(); }

TEST_F(Vector_Test, TestBadVecVecOps) {
    TestBadVecVecOps<__half>();
    TestBadVecVecOps<float>();
    TestBadVecVecOps<double>();
}

TEST_F(Vector_Test, TestBooleanEqual) {
    TestBooleanEqual<__half>();
    TestBooleanEqual<float>();
    TestBooleanEqual<double>();
}

class Vector_Sort_Test: public TestBase
{
public:

    template <typename T>
    void simple_sort_even() {

        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(3.519), static_cast<T>(8.525), static_cast<T>(3.978), static_cast<T>(8.645),
             static_cast<T>(2.798), static_cast<T>(1.477), static_cast<T>(7.021), static_cast<T>(5.689),
             static_cast<T>(6.185), static_cast<T>(6.315)}
        );

        std::vector<int> target({5, 4, 0, 2, 7, 8, 9, 6, 1, 3});
        std::vector<int> test(vec.sort_indices());

        ASSERT_EQ(target, test);

    }

    template <typename T>
    void simple_sort_odd() {

        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(7.063), static_cast<T>(8.824), static_cast<T>(5.430), static_cast<T>(5.107),
             static_cast<T>(5.478), static_cast<T>(8.819), static_cast<T>(7.995), static_cast<T>(9.787),
             static_cast<T>(4.139), static_cast<T>(8.946), static_cast<T>(4.861), static_cast<T>(1.678),
             static_cast<T>(9.176)}
        );

        std::vector<int> target({11, 8, 10, 3, 2, 4, 0, 6, 5, 1, 9, 12, 7});
        std::vector<int> test(vec.sort_indices());
    
        ASSERT_EQ(target, test);

    }

    template <typename T>
    void simple_sort_duplicates() {

        constexpr int n(7);
        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(9.433), static_cast<T>(3.950), static_cast<T>(1.776), static_cast<T>(7.016),
             static_cast<T>(1.409), static_cast<T>(1.776), static_cast<T>(7.016)}
        );

        std::vector<int> target({4, 2, 5, 1, 3, 6, 0});
        std::vector<int> test(vec.sort_indices());

        // Check that sorted element order is the same for all elements
        for (int i=0; i<n; ++i) {
            ASSERT_EQ(vec.get_elem(target[i]), vec.get_elem(test[i]));
        }

    }

    template <typename T>
    void sorted_already() {

        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4),
             static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8),
             static_cast<T>(9), static_cast<T>(10)}
        );

        std::vector<int> target({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        std::vector<int> test(vec.sort_indices());

        ASSERT_EQ(target, test);

    }

    template <typename T>
    void one_element() {

        Vector<T> vec(
            *handle_ptr,
            {static_cast<T>(1)}
        );

        std::vector<int> target({0});
        std::vector<int> test(vec.sort_indices());

        ASSERT_EQ(target, test);

    }

};

TEST_F(Vector_Sort_Test, TestSimpleSort_EvenNum) { 
    simple_sort_even<__half>();
    simple_sort_even<float>();
    simple_sort_even<double>();
}

TEST_F(Vector_Sort_Test, TestSimpleSort_OddNum) {
    simple_sort_odd<__half>();
    simple_sort_odd<float>();
    simple_sort_odd<double>();
}

TEST_F(Vector_Sort_Test, TestSimpleSort_Dupes) {
    simple_sort_duplicates<__half>();
    simple_sort_duplicates<float>();
    simple_sort_duplicates<double>();
}

TEST_F(Vector_Sort_Test, TestAlreadySorted) {
    sorted_already<__half>();
    sorted_already<float>();
    sorted_already<double>();
}

TEST_F(Vector_Sort_Test, TestOneElement) {
    one_element<__half>();
    one_element<float>();
    one_element<double>();
}
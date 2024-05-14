#include "../test.h"

#include "types/Vector/Vector.h"

class Vector_Test: public TestBase
{
public:

    template <typename T>
    void TestElementAccess() {
        
        constexpr int n(10);
        Vector<T> test_vec_n(TestBase::bundle, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }
        for (int i=0; i<n; ++i) { ASSERT_EQ(test_vec_n.get_elem(i), static_cast<T>(i*i)); }
        
        constexpr int m(18);
        Vector<T> test_vec_m(TestBase::bundle, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }
        for (int i=0; i<m; ++i) { ASSERT_EQ(test_vec_m.get_elem(i), static_cast<T>(2*i*i-m)); }
        
        constexpr int cnt(10);
        Vector<T> test_vec_cnt(TestBase::bundle, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }
        for (int i=0; i<cnt; ++i) { ASSERT_EQ(test_vec_cnt.get_elem(i), static_cast<T>(1+i)); }

    }

    void TestBadElementAccess() {

        const int m(27);
        Vector<double> vec(TestBase::bundle, m);
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.get_elem(-1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.get_elem(m); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec.set_elem(-1, 0.); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec.set_elem(m, 0.); });

    }

    template <typename T>
    void TestSlice() {

        constexpr int n(10);
        Vector<T> test_vec_n(TestBase::bundle, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        constexpr int m(18);
        Vector<T> test_vec_m(TestBase::bundle, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        constexpr int cnt(10);
        Vector<T> test_vec_cnt(TestBase::bundle, cnt);
        for (int i=0; i<cnt; ++i) { test_vec_cnt.set_elem(i, static_cast<T>(1+i)); }

        Vector<T> test_vec_cnt_2_6(test_vec_cnt.get_slice(2, 6));
        ASSERT_EQ(test_vec_cnt_2_6.rows(), 6);
        for (int i=0; i<6; ++i) { ASSERT_EQ(test_vec_cnt_2_6.get_elem(i),
                                            test_vec_cnt.get_elem(i+2)); }
        
        Vector<T> test_vec_m_1_3(test_vec_m.get_slice(1, 3));
        ASSERT_EQ(test_vec_m_1_3.rows(), 3);
        for (int i=0; i<3; ++i) { ASSERT_EQ(test_vec_m_1_3.get_elem(i),
                                            test_vec_m.get_elem(i+1)); }

        Vector<T> test_vec_n_0_4(test_vec_n.get_slice(0, 4));
        ASSERT_EQ(test_vec_n_0_4.rows(), 4);
        for (int i=0; i<4; ++i) { ASSERT_EQ(test_vec_n_0_4.get_elem(i),
                                            test_vec_n.get_elem(i)); }

        Vector<T> test_vec_m_dupe(test_vec_m.get_slice(0, m));
        ASSERT_EQ(test_vec_m_dupe.rows(), m);
        ASSERT_VECTOR_EQ(test_vec_m_dupe, test_vec_m);

        Vector<T> test_vec_get_slice_empty(test_vec_m.get_slice(0, 0));
        ASSERT_EQ(test_vec_get_slice_empty.rows(), 0);

        Vector<T> test_vec_get_slice_empty_2(test_vec_n.get_slice(1, 0));
        ASSERT_EQ(test_vec_get_slice_empty_2.rows(), 0);

    }

    template <typename T>
    void TestBadSlice() {

        constexpr int m(18);
        Vector<T> test_vec_m(TestBase::bundle, m);
        for (int i=0; i<m; ++i) { test_vec_m.set_elem(i, static_cast<T>(2*i*i-m)); }

        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.get_slice(1, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.get_slice(-1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.get_slice(m, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { test_vec_m.get_slice(0, m+1); });
        
    }

    template <typename T>
    void TestPropertyMethods() {
        
        constexpr int m(8);
        Vector<T> test_vec_m(TestBase::bundle, m);

        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);
        
        constexpr int n(16);
        Vector<T> test_vec_n(TestBase::bundle, n);
        for (int i=0; i<n; ++i) { test_vec_n.set_elem(i, static_cast<T>(i*i)); }

        ASSERT_EQ(test_vec_n.rows(), n);
        ASSERT_EQ(test_vec_n.cols(), 1);

    }

    template <typename T>
    void TestNonZeros() {
        
        constexpr int n(12);
        Vector<T> test_vec(Vector<T>::Zero(TestBase::bundle, n));
        ASSERT_EQ(test_vec.non_zeros(), 0);

        test_vec.set_elem(1, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(3, Scalar<T>(static_cast<T>(4.)));
        test_vec.set_elem(4, Scalar<T>(static_cast<T>(-1.)));
        ASSERT_EQ(test_vec.non_zeros(), 3);

        test_vec.set_elem(3, Scalar<T>(static_cast<T>(0.)));
        ASSERT_EQ(test_vec.non_zeros(), 2);

        test_vec.set_elem(2, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(3, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(5, Scalar<T>(static_cast<T>(1.)));
        ASSERT_EQ(test_vec.non_zeros(), 5);

        test_vec.set_elem(0, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(6, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(7, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(8, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(9, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(10, Scalar<T>(static_cast<T>(1.)));
        test_vec.set_elem(11, Scalar<T>(static_cast<T>(1.)));
        ASSERT_EQ(test_vec.non_zeros(), n);

    }

    void TestConstruction() {

        Vector<double> test_vec_0(TestBase::bundle, 0);
        ASSERT_EQ(test_vec_0.rows(), 0);
        ASSERT_EQ(test_vec_0.cols(), 1);

        constexpr int m(24);
        Vector<double> test_vec_m(TestBase::bundle, m);
        ASSERT_EQ(test_vec_m.rows(), m);
        ASSERT_EQ(test_vec_m.cols(), 1);

    }

    template <typename T>
    void TestListInitialization() {
        
        Vector<T> test_vec_6(
            TestBase::bundle,
            {static_cast<T>(7.), static_cast<T>(5.), static_cast<T>(3.),
             static_cast<T>(1.), static_cast<T>(6.), static_cast<T>(2.)}
        );
        ASSERT_EQ(test_vec_6.get_elem(0), static_cast<T>(7.));
        ASSERT_EQ(test_vec_6.get_elem(1), static_cast<T>(5.));
        ASSERT_EQ(test_vec_6.get_elem(2), static_cast<T>(3.));
        ASSERT_EQ(test_vec_6.get_elem(3), static_cast<T>(1.));
        ASSERT_EQ(test_vec_6.get_elem(4), static_cast<T>(6.));
        ASSERT_EQ(test_vec_6.get_elem(5), static_cast<T>(2.));

        Vector<T> test_vec_empty(TestBase::bundle, {});

    }

    template <typename T>
    void TestDynamicMemConstruction() {
    
        const int m_manual(3);
        T *h_vec_manual = static_cast<T *>(malloc(m_manual*sizeof(T)));
        h_vec_manual[0] = static_cast<T>(-5);
        h_vec_manual[1] = static_cast<T>(100);
        h_vec_manual[2] = static_cast<T>(-20);

        Vector<T> test_vec_manual(TestBase::bundle, h_vec_manual, m_manual);

        Vector<T> target_vec_manual(
            TestBase::bundle,
            {static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)}
        );

        ASSERT_VECTOR_EQ(test_vec_manual, target_vec_manual);

        free(h_vec_manual);
    
        const int m_rand(7);
        T *h_vec_rand = static_cast<T *>(malloc(m_rand*sizeof(T)));
        for (int i=0; i<m_rand; ++i) { h_vec_rand[i] = rand(); }

        Vector<T> test_vec_rand(TestBase::bundle, h_vec_rand, m_rand);

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
            TestBase::bundle,
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
            TestBase::bundle,
            {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
             static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
             static_cast<T>(rand())}
        );

        T *h_vec_rand = static_cast<T *>(malloc(m_rand*sizeof(T)));
        vec_rand.copy_data_to_ptr(h_vec_rand, m_rand);

        for (int i=0; i<m_rand; ++i) { ASSERT_EQ(h_vec_rand[i], vec_rand.get_elem(i).get_scalar()); }

        free(h_vec_rand);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_rand(10);
        Vector<double> vec_rand(TestBase::bundle, m_rand);
        double *h_vec_rand = static_cast<double *>(malloc(m_rand*sizeof(double)));
        
        auto try_row_too_small = [=]() { vec_rand.copy_data_to_ptr(h_vec_rand, m_rand-2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_small);

        auto try_row_too_large = [=]() { vec_rand.copy_data_to_ptr(h_vec_rand, m_rand+2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_large);

        free(h_vec_rand);

    }

    template <typename T>
    void TestCopyAssignment() {

        Vector<T> test_vec_empty(TestBase::bundle, {});
        Vector<T> test_vec_4(
            TestBase::bundle,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(10.)}
        );
        Vector<T> test_vec_5(
            TestBase::bundle,
            {static_cast<T>(-3.), static_cast<T>(0.), static_cast<T>(1.),
             static_cast<T>(10.), static_cast<T>(0.)}
        );
        Vector<T> test_vec_6(
            TestBase::bundle,
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
            TestBase::bundle,
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
        Vector<T> test_zero(Vector<T>::Zero(TestBase::bundle, m_zero));
        ASSERT_EQ(test_zero.rows(), m_zero);
        for (int i=0; i<m_zero; ++i) { ASSERT_EQ(test_zero.get_elem(i), static_cast<T>(0.)); }

        constexpr int m_one(15);
        Vector<T> test_ones(Vector<T>::Ones(TestBase::bundle, m_one));
        ASSERT_EQ(test_ones.rows(), m_one);
        for (int i=0; i<m_one; ++i) { ASSERT_EQ(test_ones.get_elem(i), static_cast<T>(1.)); }

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from other 5 (so only will fail if some 5 numbers in a row are exactly
        // the same))
        constexpr int m_rand(200);
        Vector<T> test_rand(Vector<T>::Random(TestBase::bundle, m_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        for (int i=2; i<m_one-2; ++i) {
            ASSERT_TRUE(
                ((test_rand.get_elem(i).get_scalar() != test_rand.get_elem(i-2).get_scalar()) ||
                 (test_rand.get_elem(i).get_scalar() != test_rand.get_elem(i-1).get_scalar()) ||
                 (test_rand.get_elem(i).get_scalar() != test_rand.get_elem(i+1).get_scalar()) ||
                 (test_rand.get_elem(i).get_scalar() != test_rand.get_elem(i+2).get_scalar()))
            );
        }

    }

    template <typename T>
    void TestScale() {

        Vector<T> vec(
            TestBase::bundle,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        Vector<T> vec_scaled_mult = vec*Scalar<T>(static_cast<T>(4));
        for (int i=0; i<5; ++i) {
            ASSERT_EQ(vec_scaled_mult.get_elem(i).get_scalar(),
                      static_cast<T>(4)*vec.get_elem(i).get_scalar());
        }

        Vector<T> vec_scaled_div = vec/Scalar<T>(static_cast<T>(10));
        for (int i=0; i<5; ++i) {
            ASSERT_EQ(vec_scaled_div.get_elem(i).get_scalar(),
            (static_cast<T>(1)/static_cast<T>(10))*vec.get_elem(i).get_scalar());
        }

    }

    template <typename T>
    void TestScaleAssignment() {

        Vector<T> orig_vec(
            TestBase::bundle,
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );

        Vector<T> vec(orig_vec);
        vec *= Scalar<T>(static_cast<T>(3));
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec.get_elem(i).get_scalar(),
                                            static_cast<T>(3)*orig_vec.get_elem(i).get_scalar()); }

        vec = orig_vec;
        vec /= Scalar<T>(static_cast<T>(5));
        for (int i=0; i<5; ++i) { ASSERT_EQ(vec.get_elem(i).get_scalar(),
                                            (static_cast<T>(1)/static_cast<T>(5))*orig_vec.get_elem(i).get_scalar()); }

    }

    template <typename T>
    void TestAddSub() {

        Vector<T> vec_1(
            TestBase::bundle,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        Vector<T> vec_2(
            TestBase::bundle,
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
            TestBase::bundle,
            {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
        );
        Vector<T> vec_2(
            TestBase::bundle,
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
            TestBase::bundle,
            {static_cast<T>(-4.), static_cast<T>(3.4), static_cast<T>(0.),
             static_cast<T>(-2.1), static_cast<T>(1.8)}
        );
        Vector<T> vec_2_dot(
            TestBase::bundle,
            {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(1.5),
             static_cast<T>(-4.5), static_cast<T>(2.)}
        );
        ASSERT_NEAR(static_cast<double>(vec_1_dot.dot(vec_2_dot).get_scalar()),
                    11.05,
                    11.05*Tol<T>::gamma(5));

        // Random
        Vector<T> vec_1_dot_r(Vector<T>::Random(TestBase::bundle, 10));
        Vector<T> vec_2_dot_r(Vector<T>::Random(TestBase::bundle, 10));
        T acc = static_cast<T>(0.);
        for (int i=0; i<10; ++i) {
            acc += vec_1_dot_r.get_elem(i).get_scalar()*vec_2_dot_r.get_elem(i).get_scalar();
        }
        ASSERT_NEAR(static_cast<double>(vec_1_dot_r.dot(vec_2_dot_r).get_scalar()),
                    static_cast<double>(acc),
                    2.*std::abs(10.*Tol<T>::gamma(10)));

    }

    template <typename T>
    void TestNorm() {

        // Pre-calculated
        Vector<T> vec_norm(
            TestBase::bundle, 
            {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
             static_cast<T>(4.), static_cast<T>(0.)}
        );
        ASSERT_NEAR(vec_norm.norm().get_scalar(),
                    static_cast<T>(9.),
                    static_cast<T>(9.)*static_cast<T>(Tol<T>::gamma(5)));

        // Random
        Vector<T> vec_norm_r(Vector<T>::Random(TestBase::bundle, 10));
        ASSERT_NEAR(vec_norm_r.norm().get_scalar(),
                    std::sqrt(vec_norm_r.dot(vec_norm_r).get_scalar()),
                    std::sqrt(vec_norm_r.dot(vec_norm_r).get_scalar())*Tol<T>::gamma(10));

    }

    template <typename T>
    void TestBadVecVecOps() {

        const int m(7);
        Vector<T> vec(Vector<T>::Random(TestBase::bundle, m));
        Vector<T> vec_too_small(Vector<T>::Random(TestBase::bundle, m-3));
        Vector<T> vec_too_large(Vector<T>::Random(TestBase::bundle, m+2));

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

        Vector<double> vec_dbl(Vector<double>::Random(TestBase::bundle, m));

        Vector<double> dbl_to_dbl(vec_dbl.cast<double>());
        ASSERT_VECTOR_EQ(dbl_to_dbl, vec_dbl);
        Vector<__half> dbl_to_hlf(vec_dbl.cast<__half>());
        ASSERT_EQ(dbl_to_hlf.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                dbl_to_hlf.get_elem(i).get_scalar(),
                static_cast<__half>(vec_dbl.get_elem(i).get_scalar()),
                min_1_mag(static_cast<__half>(vec_dbl.get_elem(i).get_scalar()))*
                    Tol<__half>::roundoff_T()
            );
        }
        Vector<float> dbl_to_sgl(vec_dbl.cast<float>());
        ASSERT_EQ(dbl_to_sgl.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                dbl_to_sgl.get_elem(i).get_scalar(),
                static_cast<float>(vec_dbl.get_elem(i).get_scalar()),
                min_1_mag(static_cast<float>(vec_dbl.get_elem(i).get_scalar()))*
                    Tol<float>::roundoff_T()
            );
        }

        Vector<float> vec_sgl(Vector<float>::Random(TestBase::bundle, m));

        Vector<float> sgl_to_sgl(vec_sgl.cast<float>());
        ASSERT_VECTOR_EQ(sgl_to_sgl, vec_sgl);
        Vector<__half> sgl_to_hlf(vec_sgl.cast<__half>());
        ASSERT_EQ(sgl_to_hlf.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                sgl_to_hlf.get_elem(i).get_scalar(),
                static_cast<__half>(vec_sgl.get_elem(i).get_scalar()),
                min_1_mag(static_cast<__half>(vec_sgl.get_elem(i).get_scalar()))*
                    Tol<__half>::roundoff_T()
            );
        }
        Vector<double> sgl_to_dbl(vec_sgl.cast<double>());
        ASSERT_EQ(sgl_to_dbl.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                sgl_to_dbl.get_elem(i).get_scalar(),
                static_cast<double>(vec_sgl.get_elem(i).get_scalar()),
                min_1_mag(static_cast<double>(vec_sgl.get_elem(i).get_scalar()))*
                    static_cast<double>(Tol<float>::roundoff_T())
            );
        }

        Vector<__half> vec_hlf(Vector<__half>::Random(TestBase::bundle, m));

        Vector<__half> hlf_to_hlf(vec_hlf.cast<__half>());
        ASSERT_VECTOR_EQ(hlf_to_hlf, vec_hlf);
        Vector<float> hlf_to_sgl(vec_hlf.cast<float>());
        ASSERT_EQ(hlf_to_sgl.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                hlf_to_sgl.get_elem(i).get_scalar(),
                static_cast<float>(vec_hlf.get_elem(i).get_scalar()),
                min_1_mag(static_cast<float>(vec_hlf.get_elem(i).get_scalar()))*static_cast<float>(Tol<__half>::roundoff_T())
            );
        }
        Vector<double> hlf_to_dbl(vec_hlf.cast<double>());
        ASSERT_EQ(hlf_to_dbl.rows(), m);
        for (int i=0; i<m; ++i) {
            ASSERT_NEAR(
                hlf_to_dbl.get_elem(i).get_scalar(),
                static_cast<double>(vec_hlf.get_elem(i).get_scalar()),
                min_1_mag(static_cast<double>(vec_hlf.get_elem(i).get_scalar()))*static_cast<double>(Tol<__half>::roundoff_T())
            );
        }

    }

    void TestBadCast() {

        auto try_bad_cast = []() {
            const int m(20);
            Vector<double> vec_dbl(Vector<double>::Random(TestBase::bundle, m));
            vec_dbl.cast<int>();
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_bad_cast);

    }

    template <typename T>
    void TestBooleanEqual() {

        Vector<T> vec_to_compare(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_same(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_diffbeg(
            TestBase::bundle,
            {static_cast<T>(1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_diffend(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(70)}
        );
        Vector<T> vec_diffmid(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(-100.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
        );
        Vector<T> vec_smaller(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101)}
        );
        Vector<T> vec_bigger(
            TestBase::bundle,
            {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
             static_cast<T>(101), static_cast<T>(-101), static_cast<T>(0)}
        );
        Vector<T> vec_empty_1(TestBase::bundle, {});
        Vector<T> vec_empty_2(TestBase::bundle, {});

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

TEST_F(Vector_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
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
    TestScale<__half>();
    TestScale<float>();
    TestScale<double>();
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

TEST_F(Vector_Test, TestBadCast) { TestBadCast(); }

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
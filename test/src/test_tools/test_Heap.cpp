#include "test.h"

#include "tools/Heap.h"

#include <map>

class PSizeHeap_Test: public TestBase
{
public:

    template <typename TPrecision>
    void FilledHeap() {

        const int n(5);
        
        heap::PSizeHeap<TPrecision> p_heap(n);

        ASSERT_EQ(p_heap.heap.size(), n);

        p_heap.push(static_cast<TPrecision>(-0.5), 4);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 1);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-0.5), 4)
        );

        p_heap.push(static_cast<TPrecision>(-1.5), 0);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 2);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-0.5), 4)
        );
        ASSERT_EQ(
            p_heap.heap[1],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-1.5), 0)
        );

        p_heap.push(static_cast<TPrecision>(-10.2), 2);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 3);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-0.5), 4)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);

        p_heap.push(static_cast<TPrecision>(0.), 1);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 4);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(0), 1)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);

        p_heap.push(static_cast<TPrecision>(5.), 3);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(0), 1)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        std::map<TPrecision, int> val_row_dict = {
            {static_cast<TPrecision>(1.5), 0},
            {static_cast<TPrecision>(0.), 1},
            {static_cast<TPrecision>(10.2), 2},
            {static_cast<TPrecision>(5.), 3},
            {static_cast<TPrecision>(0.5), 4}
        };

        for (int i=0; i<n; ++i) {
            ASSERT_EQ(
                p_heap.heap[i].row,
                val_row_dict[p_heap.heap[i].abs_val]
            );
        }

        std::map<TPrecision, TPrecision> abs_orig_dict = {
            {static_cast<TPrecision>(1.5), static_cast<TPrecision>(-1.5)},
            {static_cast<TPrecision>(0.), static_cast<TPrecision>(0.)},
            {static_cast<TPrecision>(10.2), static_cast<TPrecision>(-10.2)},
            {static_cast<TPrecision>(5.), static_cast<TPrecision>(5.)},
            {static_cast<TPrecision>(0.5), static_cast<TPrecision>(-0.5)}
        };

        for (int i=0; i<n; ++i) {
            ASSERT_EQ(
                p_heap.heap[i].orig_val,
                abs_orig_dict[p_heap.heap[i].abs_val]
            );
        }

    }

    template <typename TPrecision>
    void OverflowHeap() {

        const int n(5);
        
        heap::PSizeHeap<TPrecision> p_heap(n);

        p_heap.push(static_cast<TPrecision>(-7.5), 0);
        p_heap.push(static_cast<TPrecision>(1.), 1);
        p_heap.push(static_cast<TPrecision>(-10.2), 2);
        p_heap.push(static_cast<TPrecision>(6.), 3);
        p_heap.push(static_cast<TPrecision>(-4.5), 4);

        heap::PSizeHeap<TPrecision> p_heap_copy(p_heap);

        // Check no changes occur with the adding of a smaller value
        p_heap.push(static_cast<TPrecision>(0), 5);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        for (int i=0; i<n; ++i) {
            ASSERT_EQ(p_heap.heap[i], p_heap_copy.heap[i]);
        }

        // Check that adding similar abs value does not change heap
        p_heap.push(static_cast<TPrecision>(1.), 40);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        for (int i=0; i<n; ++i) {
            ASSERT_EQ(p_heap.heap[i], p_heap_copy.heap[i]);
        }

        // Check top gets kicked out for slightly larger values on add
        p_heap.push(static_cast<TPrecision>(-1.1), 6);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-1.1), 6)
        );
        for (int i=1; i<n; ++i) {
            ASSERT_EQ(p_heap.heap[i], p_heap_copy.heap[i]);
        }

        p_heap.push(static_cast<TPrecision>(1.2), 7);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(1.2), 7)
        );
        for (int i=1; i<n; ++i) {
            ASSERT_EQ(p_heap.heap[i], p_heap_copy.heap[i]);
        }

        // Check larger adds maintain heap property kicking out next lowest;
        p_heap.push(static_cast<TPrecision>(13), 8);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-4.5), 4)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        p_heap.push(static_cast<TPrecision>(-11), 9);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(6.), 3)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        p_heap.push(static_cast<TPrecision>(14), 10);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-7.5), 0)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        // Check with enough adds 11 becomes the next lowest
        p_heap.push(static_cast<TPrecision>(15), 11);
        p_heap.push(static_cast<TPrecision>(16.5), 12);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(-11), 9)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        // Test repeats work fine
        p_heap.push(static_cast<TPrecision>(15), 13);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(13), 8)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        // Test same absolute value as well
        p_heap.push(static_cast<TPrecision>(-15), 14);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(
            p_heap.heap[0],
            heap::ColValInfo<TPrecision>(static_cast<TPrecision>(14), 10)
        );
        ASSERT_LE(p_heap.heap[0], p_heap.heap[1]);
        ASSERT_LE(p_heap.heap[0], p_heap.heap[2]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[3]);
        ASSERT_LE(p_heap.heap[1], p_heap.heap[4]);

        // Test that the 15s were added correctly
        p_heap.push(static_cast<TPrecision>(-20), 15);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(p_heap.heap[0].abs_val, static_cast<TPrecision>(15));
        ASSERT_TRUE(
            (p_heap.heap[0].row == 11) ||
            (p_heap.heap[0].row == 13) ||
            (p_heap.heap[0].row == 14)
        );

        p_heap.push(static_cast<TPrecision>(-21), 16);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(p_heap.heap[0].abs_val, static_cast<TPrecision>(15));
        ASSERT_TRUE(
            (p_heap.heap[0].row == 11) ||
            (p_heap.heap[0].row == 13) ||
            (p_heap.heap[0].row == 14)
        );

        p_heap.push(static_cast<TPrecision>(-22), 17);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, n);
        ASSERT_EQ(p_heap.heap[0].abs_val, static_cast<TPrecision>(15));
        ASSERT_TRUE(
            (p_heap.heap[0].row == 11) ||
            (p_heap.heap[0].row == 13) ||
            (p_heap.heap[0].row == 14)
        );

    }

    void BadHeap() {
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { heap::PSizeHeap<double> p_heap(-1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { heap::PSizeHeap<double> p_heap(-5); }
        );
    }

    template <typename TPrecision>
    void EmptyHeap() {

        const int n(0);

        heap::PSizeHeap<TPrecision> p_heap(n);

        ASSERT_EQ(p_heap.heap.size(), 0);

        p_heap.push(static_cast<TPrecision>(-0.5), 4);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 0);

        p_heap.push(static_cast<TPrecision>(-1.5), 0);
        ASSERT_EQ(p_heap.heap.size(), n);
        ASSERT_EQ(p_heap.count, 0);

    }

};

TEST_F(PSizeHeap_Test, FilledHeap) {
    FilledHeap<__half>();
    FilledHeap<float>();
    FilledHeap<double>();
}

TEST_F(PSizeHeap_Test, OverflowHeap) {
    OverflowHeap<__half>();
    OverflowHeap<float>();
    OverflowHeap<double>();
}

TEST_F(PSizeHeap_Test, BadHeap) {
    BadHeap();
}

TEST_F(PSizeHeap_Test, EmptyHeap) {
    EmptyHeap<__half>();
    EmptyHeap<float>();
    EmptyHeap<double>();
}
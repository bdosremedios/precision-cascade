#ifndef HEAP_H
#define HEAP_H

#include <vector>
#include <utility>

#include "tools/AbsoluteVal.h"

namespace heap
{

template <typename T>
struct ColValInfo
{
public:

    T abs_val;
    T orig_val;
    int row;

    ColValInfo() {
        abs_val = static_cast<T>(-1);
        orig_val = static_cast<T>(-1);
        row = -1;
    }

    ColValInfo(T arg_val, int arg_row) {
        abs_val = abs_ns::abs(arg_val);
        orig_val = arg_val;
        row = arg_row;
    }

    ColValInfo & operator=(const ColValInfo &other) {
        abs_val = other.abs_val;
        orig_val = other.orig_val;
        row = other.row;
        return *this;
    }

    ColValInfo(const ColValInfo &other) {
        *this = other;
    }

    bool operator>(const ColValInfo &other) const {
        return abs_val > other.abs_val;
    }

    bool operator<=(const ColValInfo &other) const {
        return !operator>(other);
    }

    bool operator==(const ColValInfo &other) const {
        return (
            (abs_val == other.abs_val) &&
            (orig_val == other.orig_val) &&
            (row == other.row)
        );
    }


};

template <typename T>
class PSizeHeap
{
private:

    int calc_parent_ind(int curr_ind) const {
        return (curr_ind-1)/2;
    }

    int calc_child_ind_L(int curr_ind) const {
        return 2*curr_ind + 1;
    }

    void heap_swap(int i, int j) {
        ColValInfo<T> temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
    }

    void no_replace_push(ColValInfo<T> new_val) {

        heap[count] = new_val;

        int curr_ind = count;
        int parent_ind = calc_parent_ind(curr_ind);
        while ((curr_ind > 0) && (heap[parent_ind] > heap[curr_ind])) {
            heap_swap(parent_ind, curr_ind);
            curr_ind = parent_ind;
            parent_ind = calc_parent_ind(curr_ind);
        }

    }

    void replace_min_push(ColValInfo<T> new_val) {

        heap[0] = new_val;

        int curr_ind = 0;
        int child_ind_L = calc_child_ind_L(curr_ind);
        while (child_ind_L < p) {

            int min_child = child_ind_L;
            if (((child_ind_L+1) < p) && (heap[child_ind_L] > heap[child_ind_L+1])) {
                min_child = child_ind_L+1;
            }

            if (heap[curr_ind] > heap[min_child]) {
                heap_swap(curr_ind, min_child);
                curr_ind = min_child;
                child_ind_L = calc_child_ind_L(curr_ind);
            } else {
                break;
            }

        }
    
    }

public:

    const int p;
    int count = 0;
    std::vector<ColValInfo<T>> heap;

    PSizeHeap(int arg_p): p(arg_p) {
        if (p < 0) { throw std::runtime_error("PSizeHeap: invalid row size"); }
        heap.resize(p);
    }

    void push(T val, int row) {
        if (p > 0) {
            ColValInfo<T> new_val(val, row);
            if (count < p) {
                no_replace_push(new_val);
                count++;
            } else {
                if (new_val > heap[0]) {
                    replace_min_push(new_val);
                }
            }
        }
    }

    PSizeHeap(const PSizeHeap &other): PSizeHeap(other.p) {
        count = other.count;
        for (int i=0; i<count; ++i) {
            heap[i] = other.heap[i];
        }
    }

    PSizeHeap &operator=(const PSizeHeap &other) = delete;

};

}

#endif
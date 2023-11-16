#ifndef VECTOR_SORT_H
#define VECTOR_SORT_H

#include "types/types.h"

inline void swap(const int &i, const int &j, MatrixVector<int> &indices) {
    
    if (i != j) {
        int temp = indices(i);
        indices(i) = indices(j);
        indices(j) = temp;
    }

}

template <typename T>
int select_median_pivot(int i, int j, int k, const MatrixVector<int> &indices, const MatrixVector<T> &vec) {

    if ((vec(indices(i)) > vec(indices(j))) ^ (vec(indices(i)) > vec(indices(k)))) { return i; }
    else if ((vec(indices(j)) > vec(indices(i))) ^ (vec(indices(j)) > vec(indices(k)))) { return j; }
    else { return k; }

}

template <typename T>
void quicksort(MatrixVector<int> &indices, const int &beg, const int &end, const MatrixVector<T> &vec) {

    if (beg < end-1) {
        
        int pivot_ind = select_median_pivot(beg, (beg + end-1)/2, end-1, indices, vec);
        T pivot = vec(indices(pivot_ind));

        swap(pivot_ind, end-1, indices); // Move pivot to end
        pivot_ind = end-1;

        int head = beg;
        while (head < pivot_ind) {

            if (vec(indices(head)) <= pivot) {
                ++head;
            } else {
                swap(head, pivot_ind-1, indices);
                swap(pivot_ind-1, pivot_ind, indices);
                --pivot_ind;
            }

        }

        quicksort(indices, beg, pivot_ind, vec);
        quicksort(indices, pivot_ind+1, end, vec);

    }

}

template <typename T>
MatrixVector<int> sort_indices(const MatrixVector<T> &vec) {

    MatrixVector<int> indices(vec.rows());
    for (int i=0; i<vec.rows(); ++i) { indices(i) = i; }

    quicksort(indices, 0, vec.rows(), vec);

    return indices;

}

#endif
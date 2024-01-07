#ifndef VECTOR_SORT_H
#define VECTOR_SORT_H

#include "types/types.h"

inline void swap(const int &i, const int &j, MatrixVector<int> &indices) {
    
    if (i != j) {
        int temp = indices.get_elem(i);
        indices.set_elem(i, indices.get_elem(j));
        indices.set_elem(j, temp);
    }

}

template <typename T>
int select_median_pivot(int i, int j, int k, const MatrixVector<int> &indices, const MatrixVector<T> &vec) {

    if ((vec.get_elem(indices.get_elem(i)) > vec.get_elem(indices.get_elem(j))) ^
        (vec.get_elem(indices.get_elem(i)) > vec.get_elem(indices.get_elem(k)))) { return i; }
    else if ((vec.get_elem(indices.get_elem(j)) > vec.get_elem(indices.get_elem(i))) ^
             (vec.get_elem(indices.get_elem(j)) > vec.get_elem(indices.get_elem(k)))) { return j; }
    else { return k; }

}

template <typename T>
void quicksort(MatrixVector<int> &indices, const int &beg, const int &end, const MatrixVector<T> &vec) {

    if (beg < end-1) {
        
        int pivot_ind = select_median_pivot(beg, (beg + end-1)/2, end-1, indices, vec);
        T pivot = vec.get_elem(indices.get_elem(pivot_ind));

        swap(pivot_ind, end-1, indices); // Move pivot to end
        pivot_ind = end-1;

        int head = beg;
        while (head < pivot_ind) {

            if (vec.get_elem(indices.get_elem(head)) <= pivot) {
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
    for (int i=0; i<vec.rows(); ++i) { indices.set_elem(i, i); }

    quicksort(indices, 0, vec.rows(), vec);

    return indices;

}

#endif
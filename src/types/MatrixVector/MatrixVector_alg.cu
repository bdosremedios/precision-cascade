// #include "types/MatrixVector.h"

// inline void swap(int *indices, const int &i, const int &j) {
//     if (i != j) {
//         int temp = indices[i];
//         indices[i] = indices[j];
//         indices[j] = temp;
//     }
// }

// template <typename T>
// int select_median_pivot(const int *indices, const T *h_vec, const int &i, const int &j, const int &k) {
//     if ((h_vec[indices[i]] > h_vec[indices[j]]) ^ ((h_vec[indices[i]]) > h_vec[indices[k]])) { return i; }
//     else if ((h_vec[indices[j]] > h_vec[indices[i]]) ^ (h_vec[indices[j]] > h_vec[indices[k]])) { return j; }
//     else { return k; }
// }

// template <typename T>
// void quicksort(int *indices, const T *h_vec, const int &beg, const int &end) {

//     if (beg < end-1) {
        
//         int pivot_ind = select_median_pivot(indices, h_vec, beg, (beg + end-1)/2, end-1);
//         T pivot = h_vec[indices[pivot_ind]];

//         swap(indices, pivot_ind, end-1); // Move pivot to end
//         pivot_ind = end-1;

//         int head = beg;
//         while (head < pivot_ind) {
//             if (h_vec[indices[head]] <= pivot) {
//                 ++head;
//             } else {
//                 swap(indices, head, pivot_ind-1);
//                 swap(indices, pivot_ind-1, pivot_ind);
//                 --pivot_ind;
//             }
//         }

//         quicksort(indices, h_vec, beg, pivot_ind);
//         quicksort(indices, h_vec, pivot_ind+1, end);

//     }

// }

// template <typename T>
// MatrixVector<int> MatrixVector<T>::sort_indices() const {
    
//     int *h_indices = static_cast<int *>(malloc(m_rows*sizeof(int)));
//     int *h_vec = static_cast<T *>(malloc(m_rows*sizeof(T)));

//     for (int i=0; i<m_rows; ++i) { h_indices[i] = i; }
//     cublasGetVector(m_rows, sizeof(T), d_vec, 1, h_vec, 1);

//     quicksort(h_indices, h_vec, 0, m_rows);

//     MatrixVector<int> indices(handle, h_indices, m_rows);

//     free(h_indices);
//     free(h_vec);

//     return indices;

// }
#ifndef SORT_H
#define SORT_H

namespace sort
{

namespace
{

template <typename T>
int select_median_ind(int i, int j, int k, T *arr) {
    if ((arr[i] > arr[j]) ^ (arr[i] > arr[k])) {
        return i;
    } else if ((arr[j] > arr[i]) ^ (arr[j] > arr[k])) {
        return j;
    } else {
        return k;
    }
}

template <typename T, typename W>
void passengered_swap(int i, int j, T *arr_1, W *arr_2) {
    T temp_1 = arr_1[i];
    W temp_2 = arr_2[i];
    arr_1[i] = arr_1[j];
    arr_2[i] = arr_2[j];
    arr_1[j] = temp_1;
    arr_2[j] = temp_2;
}

}

template <typename T, typename W>
void in_place_passengered_sort(int beg, int end, T *sorting_arr, W *passenger_arr) {

    if (beg < end-1) {

        int pivot_ind = select_median_ind(
            beg, (beg + end-1)/2, end-1, sorting_arr
        );
        T pivot = sorting_arr[pivot_ind];

        passengered_swap(pivot_ind, end-1, sorting_arr, passenger_arr); // Move pivot to end
        pivot_ind = end-1;

        int head = beg;
        while (head < pivot_ind) {
            if (sorting_arr[head] <= pivot) {
                ++head;
            } else {
                passengered_swap(head, pivot_ind-1, sorting_arr, passenger_arr);
                passengered_swap(pivot_ind-1, pivot_ind, sorting_arr, passenger_arr);
                --pivot_ind;
            }
        }

        in_place_passengered_sort(beg, pivot_ind, sorting_arr, passenger_arr);
        in_place_passengered_sort(pivot_ind+1, end, sorting_arr, passenger_arr);

    }

}
    
}

#endif
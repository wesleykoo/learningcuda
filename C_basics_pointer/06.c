#include <stdio.h>

int main() {
    int arr[] = {1,2,3,4};
    int arr2[] = {5,6,7,8};
    int *ptr = arr;
    int *ptr2 = arr2;
    int *matrix[] = {ptr, ptr2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%d", *matrix[i]++);
        }
        printf("\n");
    }
}

// output
// 1234
// 5678
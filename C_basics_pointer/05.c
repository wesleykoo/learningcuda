#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};
    int *ptr = arr; // pointer to the first element of the array

    for (int i = 0; i < 5; i++) {
        printf("Value of Position %d: %d\n", i, *ptr);
        printf("Address of Position %d: %p\n", i, ptr);
        ptr++;
    }

}

// output:
// Value of Position 0: 12
// Address of Position 0: 0x7ffe1e202520
// Value of Position 1: 24
// Address of Position 1: 0x7ffe1e202524
// Value of Position 2: 36
// Address of Position 2: 0x7ffe1e202528
// Value of Position 3: 48
// Address of Position 3: 0x7ffe1e20252c
// Value of Position 4: 60
// Address of Position 4: 0x7ffe1e202530
#include <stdio.h>

int main() {
    int value = 42;
    int *ptr = &value;
    int **ptr2 = &ptr;
    int ***ptr3 = &ptr2;

    printf("Value: %d\n", value);
    printf("Address of value: %p\n", ptr);
    printf("Address of ptr: %p\n", ptr2);
    printf("Address of ptr2: %p\n", ptr3);
    printf("Value at ptr3: %d\n", ***ptr3);
}
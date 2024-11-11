#include <stdio.h>

// & "address of" operator
// * "value at address, dereference" operator

int main() {
    int x = 10;
    int *ptr = &x;
    printf("Address of x: %p\n", ptr); // %p is pointer
    printf("Value of x: %d\n", *ptr); // *ptr is value at address
}
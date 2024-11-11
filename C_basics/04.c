// purpose: NULL pointer init and safe dereferencing
// key points:
// 1. Initialize pointer with NULL when they don't yet point and valid data.
// 2. Check pointers for NULL before dereferencing them.
// 3. NULL check allows graceful handling of uninitialized or failed allocations.

#include <stdio.h>
#include <stdlib.h>

int main() {
    // initialize pointer with NULL
    int *ptr = NULL;
    printf("1. Initial value of ptr: %p\n", (void *)ptr);

    // check for NULL before dereferencing
    if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference.\n");
    } else {
        printf("2. Value at ptr: %d\n", *ptr);
    }

    // allocate memory
    ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        printf("3. Memory allocation failed.\n");
        return 1;
    }

    printf("4. After allocation, ptr address: %p\n", (void *)ptr); // need to check
    
    // safe to use ptr after NULL check and allocation
    *ptr = 42;
    printf("5. Value at ptr after initialization: %d\n", *ptr);

    // free allocated memory
    free(ptr);
    ptr = NULL; // good practice to set pointer to NULL after free

    printf("6. After free, ptr address: %p\n", (void *)ptr);

    // Demonstrate safe dereferencing after freeing
    if (ptr == NULL) {
        printf("7. ptr is NULL, safe to avoid use after free.\n");
    }

    return 0;
}

// output:
// 1. Initial value of ptr: (nil)
// 2. ptr is NULL, cannot dereference.
// 4. After allocation, ptr address: 0x607ec36076b0
// 5. Value at ptr after initialization: 42
// 6. After free, ptr address: (nil)
// 7. ptr is NULL, safe to avoid use after free.
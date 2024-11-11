# include <stdio.h>

// void pointer: used when we don't know the data type of the memory address

int main() {
    int num = 10;
    float fnum = 3.14;
    void *vptr;

    vptr = &num;
    printf("Value of num: %d\n", *(int *)vptr);

    vptr = &fnum;
    printf("Value of fnum: %f\n", *(float *)vptr);

}
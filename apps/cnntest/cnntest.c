#include <stdio.h>
#include "cnninst.h"

int main() {
  printf("test act\n");
  uint16_t A[8] = {1, 10, 200, 0};
  uint16_t B[8] = {0};

  assert(((uint64_t)A % sizeof(uint64_t)) == 0);
  assert(((uint64_t)B % sizeof(uint64_t)) == 0);

  uint64_t vwidth;
  LoadV_Width((uint64_t)&vwidth);

  int size = 2;
  uint64_t zero;
  uint64_t *a_ptr;
  uint64_t *b_ptr;

  zero = 0x0010001000100010;
  a_ptr = (uint64_t *)A;
  b_ptr = (uint64_t *)B;
  for (int i=0; i<size; i++) {
    b_ptr[i] = Act(*a_ptr, zero);
    a_ptr ++;
  }

  for (int i=0; i<8; i++) {
    printf("%d ", B[i]);
  }

  return 0;
}

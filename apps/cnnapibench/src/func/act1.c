#include "benchmark.h"
#include "cnnapi.h"
#include "cnnapi_common.h"
#include "cnnapi_stdins.h"

static image_t *A;     // input
static image_t *B;     // cnn output
static image_t *C;     // std output
static int w = 20;
static int h = 20;
static uint16_t zero_point = 6;

static int test_pass;

void bench_act1_prepare() {
  bench_srand(1);
  A = RandomInitImage_SC(w, h, 4, 1);
  test_pass = 1;
}

void bench_act1_run() {
  B = Activation_SC(A, "relu", zero_point);
  C = StdIns_Activation_SC(A, "relu", zero_point);
}

int bench_act1_validate() {
  if (B->width == C->width && B->height == C->height && B->vwidth == C->vwidth) {
    for (int j=0; j<B->width; j++) {
      for (int i=0; i<B->height; i++) {
        uint16_t temp1 = get_main_value((uint64_t *)(B->addr), j * B->height + i, B->vwidth);
        uint16_t temp2 = get_main_value((uint64_t *)(C->addr), j * B->height + i, B->vwidth);
        if (temp1 != temp2) {
          printf("  act error: i=%d, j=%d, act_res=%d, std_res=%d\n", i, j, temp1, temp2);
          printf("  %d\n", get_main_value((uint64_t *)(A->addr), j * A->height + i, A->vwidth));
          test_pass = 0;
        }
      }
    }
  }
  else {
    test_pass = 0;
  }

  if (test_pass == 1) {
    printf("end: pass!!!\n");
  }
  else {
    printf("end: fail\n");
  }
  free(A);
  free(B);
  free(C);
  return (setting->checksum == 0x00000021) && test_pass;
}

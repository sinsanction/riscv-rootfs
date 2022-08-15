#include "benchmark.h"
#include "cnnapi_v2.h"
#include "cnnapi_common.h"
#include "cnnapi_v2_stdins.h"

static image_mp_t *A;     // input
static image_mp_t *B;     // cnn output
static image_mp_t *C;     // std output
static int w = 20;
static int h = 20;
static uint16_t *zero_point;

static int test_pass;

void bench_act1_v2_prepare() {
  bench_srand(1);
  A = RandomInitImage_MP_SC(w, h);
  zero_point = (uint16_t *)malloc(sizeof(uint16_t) * w);
  for (int i=0; i<w; i++) {
    zero_point[i] = bench_rand() & 0xf;
  }
  test_pass = 1;
}

void bench_act1_v2_run() {
  B = Activation_MP_SC(A, "relu", zero_point);
  C = StdIns_Activation_MP_SC(A, "relu", zero_point);
}

int bench_act1_v2_validate() {
  if (B->width == C->width && B->height == C->height) {
    for (int j=0; j<B->width; j++) {
      for (int i=0; i<B->height; i++) {
        uint16_t temp1 = get_main_value((uint64_t *)(B->addr[j]), i, B->vwidth[j]);
        uint16_t temp2 = get_main_value((uint64_t *)(C->addr[j]), i, C->vwidth[j]);
        if (temp1 != temp2) {
          printf("  act error: i=%d, j=%d, act_res=%d, std_res=%d\n", i, j, temp1, temp2);
          printf("  %d\n", get_main_value((uint64_t *)(A->addr[j]), i, A->vwidth[j]));
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
  return (setting->checksum == 0x00020021) && test_pass;
}

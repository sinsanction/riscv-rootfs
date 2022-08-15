#include "benchmark.h"
#include "cnnapi_v2.h"
#include "cnnapi_common.h"
#include "cnnapi_v2_stdins.h"

static image_mp_t *A;     // input
static image_mp_t *B;     // cnn output
static image_mp_t *C;     // std output
static int w = 28;
static int h = 28;
static int k = 4;
static int strides = 2;

static int test_pass;

void bench_pool2_v2_prepare() {
  bench_srand(1);
  A = RandomInitImage_MP_SC(w, h);
  test_pass = 1;
}

void bench_pool2_v2_run() {
  B = AvgPooling_MP_SC(A, k, strides);
  C = StdIns_AvgPooling_MP_SC(A, k, strides);
}

int bench_pool2_v2_validate() {
  if (B->width == C->width && B->height == C->height) {
    for (int j=0; j<B->width; j++) {
      for (int i=0; i<B->height; i++) {
        uint16_t temp1 = get_main_value((uint64_t *)(B->addr[j]), i, B->vwidth[j]);
        uint16_t temp2 = get_main_value((uint64_t *)(C->addr[j]), i, C->vwidth[j]);
        if (temp1 != temp2) {
          printf("  pool error: i=%d, j=%d, pool_res=%d, std_res=%d\n", i, j, temp1, temp2);
          for (int si=0; si<k; si++) {
            for (int sj=0; sj<k; sj++) {
              printf("  %d", get_main_value((uint64_t *)(A->addr[j * strides + sj]), (i * strides + si), A->vwidth[j * strides + sj]));
            }
          }
          printf("\n");
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
  return (setting->checksum == 0x00020012) && test_pass;
}

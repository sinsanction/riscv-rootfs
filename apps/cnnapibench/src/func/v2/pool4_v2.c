#include "benchmark.h"
#include "cnnapi_v2.h"
#include "cnnapi_common.h"
#include "cnnapi_v2_stdins.h"

static image_mp_mc_t *A;     // input
static image_mp_mc_t *B;     // cnn output
static image_mp_mc_t *C;     // std output
static int w = 20;
static int h = 20;
static int k = 3;
static int strides = 1;
static int channel = 3;

static int test_pass;

void bench_pool4_v2_prepare() {
  bench_srand(1);
  A = RandomInitImage_MP(w, h, channel);
  test_pass = 1;
}

void bench_pool4_v2_run() {
  B = MaxPooling_MP(A, k, strides);
  C = StdIns_MaxPooling_MP(A, k, strides);
}

int bench_pool4_v2_validate() {
  if (B->width == C->width && B->height == C->height && B->channel == C->channel) {
    for (int c=0; c<B->channel; c++) {
      for (int j=0; j<B->width; j++) {
        for (int i=0; i<B->height; i++) {
          uint16_t temp1 = get_main_value((uint64_t *)(B->img[c]->addr[j]), i, B->img[c]->vwidth[j]);
          uint16_t temp2 = get_main_value((uint64_t *)(C->img[c]->addr[j]), i, C->img[c]->vwidth[j]);
          if (temp1 != temp2) {
            printf("  pool error: c=%d, i=%d, j=%d, pool_res=%d, std_res=%d\n", c, i, j, temp1, temp2);
            test_pass = 0;
          }
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
  return (setting->checksum == 0x00020014) && test_pass;
}

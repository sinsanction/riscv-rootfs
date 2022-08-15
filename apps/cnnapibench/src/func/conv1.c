#include "benchmark.h"
#include "cnnapi.h"
#include "cnnapi_common.h"
#include "cnnapi_stdins.h"

static image_t *A;     // input
static image_t *B;     // cnn output
static image_t *C;     // std output
static kernel_t *kernel;
static int w = 30;
static int h = 30;
static int k = 5;
static int strides = 5;
static out_scale_t out_scale;

static int test_pass;

void bench_conv1_prepare() {
  bench_srand(1);
  A = RandomInitImage_SC(w, h, 8, 1);
  kernel = RandomInitKernel_SC(k, 8);
  out_scale.scale = A->scale / 2;
  out_scale.zero_point = 0;
  test_pass = 1;
  SetOutputKernel_SC(kernel);
}

void bench_conv1_run() {
  B = Convolution_SC(A, kernel, strides, &out_scale);
  C = StdIns_Convolution_SC(A, kernel, strides, &out_scale);
}

int bench_conv1_validate() {
  if (B->width == C->width && B->height == C->height && B->vwidth == C->vwidth) {
    for (int j=0; j<B->width; j++) {
      for (int i=0; i<B->height; i++) {
        uint16_t temp1 = get_main_value((uint64_t *)(B->addr), j * B->height + i, B->vwidth);
        uint16_t temp2 = get_main_value((uint64_t *)(C->addr), j * B->height + i, B->vwidth);
        if (temp1 != temp2) {
          printf("  conv error: i=%d, j=%d, conv_res=%d, std_res=%d\n", i, j, temp1, temp2);
          for (int si=0; si<k; si++) {
            for (int sj=0; sj<k; sj++) {
              printf("  %d", get_main_value((uint64_t *)(A->addr), (j * strides + sj) * h + (i * strides + si), A->vwidth));
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
  free(kernel);
  return (setting->checksum == 0x00000001) && test_pass;
}

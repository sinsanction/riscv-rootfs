#include "benchmark.h"
#include "cnnapi.h"
#include "cnnapi_common.h"
#include "cnnapi_stdins.h"

static image_mc_t *input;
static kernel_mc_t *c1_ker;
static kernel_mc_t *c3_ker;
static fc_filter_t *fc_filter1;
static fc_filter_t *fc_filter2;
static fc_filter_t *fc_filter_out;
static image_t *output;
static out_scale_mc_t c1_out_scale;
static out_scale_mc_t c3_out_scale;
static out_scale_t fc1_pre_out_scale;
static out_scale_t fc1_out_scale;
static out_scale_t fc2_out_scale;
static out_scale_t fc3_out_scale;

static int test_pass;

void bench_lenet5_perf_prepare() {
  bench_srand(1);
  test_pass = 1;

  // input 32x32x1
  input = RandomInitImage(32, 32, 8, 1);
  //kernel size = 5*5  strides = 1  num = 6
  c1_ker = RandomInitKernel(5, 8, 1, 6);
  c1_out_scale.channel = 6;
  c1_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 6);
  for (int i=0; i<6; i++) {
    c1_out_scale.scale[i].scale = input->img[0]->scale / 2;
    c1_out_scale.scale[i].zero_point = 0;
  }
  //kernel size = 5*5  strides = 1  num = 16
  c3_ker = RandomInitKernel(5, 8, 6, 16);
  c3_out_scale.channel = 16;
  c3_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 16);
  for (int i=0; i<16; i++) {
    c3_out_scale.scale[i].scale = input->img[0]->scale / 4;
    c3_out_scale.scale[i].zero_point = 0;
  }

  fc_filter1 = RandomInitFcFilterArray(1, 400, 8, 120);
  fc_filter2 = RandomInitFcFilterArray(1, 120, 8, 84);
  fc_filter_out = RandomInitFcFilterArray(1, 84, 8, 10);
  fc1_pre_out_scale.scale = input->img[0]->scale / 4;
  fc1_pre_out_scale.zero_point = 0;
  fc1_out_scale.scale = input->img[0]->scale / 8;
  fc1_out_scale.zero_point = 0;
  fc2_out_scale.scale = input->img[0]->scale / 16;
  fc2_out_scale.zero_point = 0;
  fc3_out_scale.scale = input->img[0]->scale / 16;
  fc3_out_scale.zero_point = 0;
}

void bench_lenet5_perf_run() {

    // layer1 c1
    image_mc_t *c1 = Convolution(input, c1_ker, 1, &c1_out_scale);
    //outsize 28*28*6

    // layer2 s2
    //pool size = 2*2  strides = 2
    image_mc_t *s2 = MaxPooling(c1, 2, 2);
    //outsize 14*14*6

    // layer3 c3
    image_mc_t *c3 = Convolution(s2, c3_ker, 1, &c3_out_scale);
    //outsize 10*10*16

    // layer4 s4
    //pool size = 2*2  strides = 2
    image_mc_t *s4 = MaxPooling(c3, 2, 2);
    //outsize 5*5*16

    // layer5 fc1
    image_t *fc1_pre = Flatten(s4, &fc1_pre_out_scale, 0x40);
    //outsize 1*400
    image_t *fc1 = Dense(fc1_pre, fc_filter1, 120, &fc1_out_scale);
    //outsize 1*120

    // layer6 fc2
    image_t *fc2 = Dense(fc1, fc_filter2, 84, &fc2_out_scale);
    //outsize 1*84

    // ouput 1*10
    output = Dense(fc2, fc_filter_out, 10, &fc3_out_scale);
    SetOutput_SC(output);
}

int bench_lenet5_perf_validate() {
  if (test_pass == 1) {
    printf("end: pass!!!\n");
  }
  else {
    printf("end: fail\n");
    SetOutput(input);
  }
  free(input);
  free(output);
  return (setting->checksum == 0x00030003) && test_pass;
}

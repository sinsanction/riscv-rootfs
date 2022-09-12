#include "benchmark.h"
#include "cnnapi.h"
#include "cnnapi_common.h"
#include "cnnapi_stdins.h"
#include "lenet5_data.h"

static image_mc_t *input;
static kernel_mc_t *c1_ker;
static kernel_mc_t *c3_ker;
static fc_filter_t *fc_filter1;
static fc_filter_t *fc_filter2;
static fc_filter_t *fc_filter3;
static image_t *output;
static image_t *output_std;

static out_scale_mc_t input_out_scale;
static out_scale_mc_t c1_out_scale;
static out_scale_mc_t s2_out_scale;
static out_scale_mc_t c3_out_scale;
static out_scale_mc_t s4_out_scale;
static out_scale_t fc1_out_scale;
static out_scale_t fc2_out_scale;
static out_scale_t fc3_out_scale;

static int test_pass;

void bench_lenet5_real_func_prepare() {
  bench_srand(1);
  test_pass = 1;
  remap_data();
  remap_image0();
  remap_image1();
  remap_image2();
  remap_image3();
  remap_image4();
  remap_image5();
  remap_image6();
  remap_image7();
  remap_image8();
  remap_image9();

  input_out_scale.channel = 1;
  input_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 1);
  input_out_scale.scale[0].scale = 47;
  input_out_scale.scale[0].zero_point = -94 + 128;

  //kernel size = 5*5  strides = 1  num = 6
  c1_ker = InitKernel(5, 8, 1, 6, 189, C1_Bias, C1_Kernel);

  c1_out_scale.channel = 6;
  c1_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 6);
  for (int i=0; i<6; i++) {
    c1_out_scale.scale[i].scale = 28;
    c1_out_scale.scale[i].zero_point = -128 + 128;
  }

  //pool size = 2*2
  s2_out_scale.channel = 6;
  s2_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 6);
  for (int i=0; i<6; i++) {
    s2_out_scale.scale[i].scale = 25;
    s2_out_scale.scale[i].zero_point = -128 + 128;
  }

  //kernel size = 5*5  strides = 1  num = 16
  c3_ker = InitKernel(5, 8, 6, 16, 246, C3_Bias, C3_Kernel);

  c3_out_scale.channel = 16;
  c3_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 16);
  for (int i=0; i<16; i++) {
    c3_out_scale.scale[i].scale = 12;
    c3_out_scale.scale[i].zero_point = 8 + 128;
  }

  //pool size = 2*2
  s4_out_scale.channel = 16;
  s4_out_scale.scale = (out_scale_t *)malloc(sizeof(out_scale_t) * 16);
  for (int i=0; i<16; i++) {
    s4_out_scale.scale[i].scale = 13;
    s4_out_scale.scale[i].zero_point = -5 + 128;
  }

  //fc1 256->120
  fc_filter1 = InitFcFilterArray(1, 256, 8, 120, 285, FC1_Bias, FC1_Filter);
  fc1_out_scale.scale = 16;
  fc1_out_scale.zero_point = -127 + 128;
  
  //fc2 120->84
  fc_filter2 = InitFcFilterArray(1, 120, 8, 84, 319, FC2_Bias, FC2_Filter);
  fc2_out_scale.scale = 14;
  fc2_out_scale.zero_point = -127 + 128;

  //fc3 84->10
  fc_filter3 = InitFcFilterArray(1, 84, 8, 10, 178, FC3_Bias, FC3_Filter);
  fc3_out_scale.scale = 8;
  fc3_out_scale.zero_point = 12 + 128;
}

static void lenet5() {
  // input
  Rescale(input, &input_out_scale);

  // layer1 c1
  image_mc_t *c1 = Convolution(input, c1_ker, 1, &c1_out_scale);
  //outsize 24*24*6

  // layer2 s2
  //pool size = 2*2  strides = 2
  image_mc_t *s2 = MaxPooling(c1, 2, 2);
  Rescale(s2, &s2_out_scale);
  //outsize 12*12*6

  // layer3 c3
  image_mc_t *c3 = Convolution(s2, c3_ker, 1, &c3_out_scale);
  //outsize 8*8*16

  // layer4 s4
  //pool size = 2*2  strides = 2
  image_mc_t *s4 = MaxPooling(c3, 2, 2);
  Rescale(s4, &s4_out_scale);
  //outsize 4*4*16

  image_t *fc1_pre = Flatten(s4);
  //outsize 1*256

  // layer5 fc1
  image_t *fc1 = Dense(fc1_pre, fc_filter1, 120, &fc1_out_scale);
  //outsize 1*120

  // layer6 fc2
  image_t *fc2 = Dense(fc1, fc_filter2, 84, &fc2_out_scale);
  //outsize 1*84

  // ouput 1*10
  output = Dense(fc2, fc_filter3, 10, &fc3_out_scale);

  // free
  free(c1);
  free(s2);
  free(c3);
  free(s4);
  free(fc1_pre);
  free(fc1);
  free(fc2);
}

static void lenet5_std() {
  // input
  Rescale(input, &input_out_scale);

  // layer1 c1
  image_mc_t *c1 = StdIns_Convolution(input, c1_ker, 1, &c1_out_scale);
  //outsize 24*24*6

  // layer2 s2
  //pool size = 2*2  strides = 2
  image_mc_t *s2 = StdIns_MaxPooling(c1, 2, 2);
  Rescale(s2, &s2_out_scale);
  //outsize 12*12*6

  // layer3 c3
  image_mc_t *c3 = StdIns_Convolution(s2, c3_ker, 1, &c3_out_scale);
  //outsize 8*8*16

  // layer4 s4
  //pool size = 2*2  strides = 2
  image_mc_t *s4 = StdIns_MaxPooling(c3, 2, 2);
  Rescale(s4, &s4_out_scale);
  //outsize 4*4*16

  image_t *fc1_pre = Flatten(s4);
  //outsize 1*256

  // layer5 fc1
  image_t *fc1 = Dense(fc1_pre, fc_filter1, 120, &fc1_out_scale);
  //outsize 1*120

  // layer6 fc2
  image_t *fc2 = Dense(fc1, fc_filter2, 84, &fc2_out_scale);
  //outsize 1*84

  // ouput 1*10
  output_std = Dense(fc2, fc_filter3, 10, &fc3_out_scale);

  // free
  free(c1);
  free(s2);
  free(c3);
  free(s4);
  free(fc1_pre);
  free(fc1);
  free(fc2);
}

static void output_res(int num) {
  int max_value = 0;
  printf(" cnn: ");
  for (int i=0; i<output->height; i++) {
    uint16_t res_value = get_main_value((uint64_t *)(output->addr), i, output->vwidth);
    max_value = (max_value < res_value) ? res_value : max_value;
    printf(" %d ", res_value);
  }
  printf(" max: %d ", max_value);
  printf(" number: ");
  for (int i=0; i<output->height; i++) {
    if (get_main_value((uint64_t *)(output->addr), i, output->vwidth) == max_value) {
      printf("%d ", i);
    }
  }
  printf(" ref: %d\n", num);

  printf(" std: ");
  max_value = 0;
  for (int i=0; i<output_std->height; i++) {
    uint16_t res_value = get_main_value((uint64_t *)(output_std->addr), i, output_std->vwidth);
    max_value = (max_value < res_value) ? res_value : max_value;
    printf(" %d ", res_value);
  }
  printf(" max: %d ", max_value);
  printf(" number: ");
  for (int i=0; i<output_std->height; i++) {
    if (get_main_value((uint64_t *)(output_std->addr), i, output_std->vwidth) == max_value) {
      printf("%d ", i);
    }
  }
  printf(" ref: %d\n", num);
}

static void output_cmp() {
  if (output->width == output_std->width && output->height == output_std->height && output->vwidth == output_std->vwidth) {
    for (int j=0; j<output->width; j++) {
      for (int i=0; i<output->height; i++) {
        uint16_t temp1 = get_main_value((uint64_t *)(output->addr), j * output->height + i, output->vwidth);
        uint16_t temp2 = get_main_value((uint64_t *)(output_std->addr), j * output->height + i, output->vwidth);
        if (temp1 != temp2) {
          printf("  lenet error: i=%d, j=%d, cnn_res=%d, std_res=%d\n", i, j, temp1, temp2);
          test_pass = 0;
        }
      }
    }
  }
  else {
    test_pass = 0;
  }
}

void bench_lenet5_real_func_run() {
  int test_num[10] = { 1, 1, 1, 1, 1,  1, 1, 1, 1, 1 };
  int test_id[10][10] = { {1}, {0}, {1}, {0}, {1},  {0}, {1}, {0}, {1}, {0} };
  for (int i=0; i<=9; i++) {
    printf("Number #%d: \n", i);
    for (int j=0; j<test_num[i]; j++) {
      // input 28x28x1
      input = InitImage(28, 28, 8, 1, 0, 255, 0, Input_Image[i]+(28*28)*test_id[i][j]);
      lenet5();
      free(input);

      input = InitImage(28, 28, 8, 1, 0, 255, 0, Input_Image[i]+(28*28)*test_id[i][j]);
      lenet5_std();
      free(input);

      output_cmp();
      output_res(i);
      free(output);
      free(output_std);
    }
    printf("\n");
  }
  printf("==================================================\n");
}

int bench_lenet5_real_func_validate() {
  if (test_pass == 1) {
    printf("end: pass!!!\n");
  }
  else {
    printf("end: fail\n");
  }
  return (setting->checksum == 0x00040001) && test_pass;
}

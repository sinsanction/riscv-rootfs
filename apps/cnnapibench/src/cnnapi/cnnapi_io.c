#include "cnnapi.h"
#include "cnnapi_common.h"


// IO
image_t *RandomInitImage_SC(uint32_t width, uint32_t height, uint32_t bits, uint8_t order) {

  image_t *img = (image_t *)malloc(sizeof(image_t));
  img->width = width;
  img->height = height;
  img->order = order;
  img->zero_point = 0;

  if (bits == 16) {
    img->vwidth = 0x80;
    img->scale = 32768;
  }
  else if (bits == 8) {
    img->vwidth = 0x40;
    img->scale = 128;
  }
  else if (bits == 4) {
    img->vwidth = 0x20;
    img->scale = 8;
  }
  else if (bits == 2) {
    img->vwidth = 0x10;
    img->scale = 2;
  }
  else {
    free(img);
    return NULL;
  }

  int size = round_up_div(width * height * bits, 64);
  uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  uint16_t *data = (uint16_t *)img_data;
  for (int i=0; i<size*4; i++) {
    // RAND_MAX assumed to be 32767
    data[i] = rand();
  }
  img->addr = (void *)img_data;

  return img;
}

kernel_t *RandomInitKernel_SC(uint32_t k, uint32_t bits) {

  kernel_t *kernel = (kernel_t *)malloc(sizeof(kernel_t));
  kernel->size = k;
  kernel->bias = 0;

  if (bits == 8) {
    kernel->vwidth = 0x8;
    kernel->scale = 128;
  }
  else if (bits == 4) {
    kernel->vwidth = 0x4;
    kernel->scale = 8;
  }
  else if (bits == 2) {
    kernel->vwidth = 0x2;
    kernel->scale = 2;
  }
  else if (bits == 1) {
    kernel->vwidth = 0x1;
    kernel->scale = 1;
  }
  else {
    free(kernel);
    return NULL;
  }

  int size = round_up_div(k * k * bits, 64);
  uint64_t *kernel_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  uint16_t *data = (uint16_t *)kernel_data;
  for (int i=0; i<size*4; i++) {
    // RAND_MAX assumed to be 32767
    data[i] = rand();
  }
  kernel->addr = (void *)kernel_data;

  return kernel;
}

image_mc_t *RandomInitImage(uint32_t width, uint32_t height, uint32_t bits, uint16_t channel) {
  
  image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
  img_mc->width = width;
  img_mc->height = height;
  img_mc->channel = channel;
  img_mc->order = 1;

  if (! ((bits == 16) || (bits == 8) || (bits == 4) || (bits == 2)) ) {
    free(img_mc);
    return NULL;
  }

  for (int i=0; i<channel; i++) {
    img_mc->img[i] = RandomInitImage_SC(width, height, bits, 1);
  }

  return img_mc;
}

kernel_mc_t *RandomInitKernel(uint32_t k, uint32_t bits, uint16_t in_channel, uint16_t out_channel) {

  kernel_mc_t *ker_mc = (kernel_mc_t *)malloc(sizeof(kernel_mc_t));
  ker_mc->size = k;
  ker_mc->in_channel = in_channel;
  ker_mc->out_channel = out_channel;

  if (! ((bits == 8) || (bits == 4) || (bits == 2) || (bits == 1)) ) {
    free(ker_mc);
    return NULL;
  }

  for (int i=0; i<in_channel*out_channel; i++) {
    ker_mc->ker[i] = RandomInitKernel_SC(k, bits);
  }

  return ker_mc;
}

fc_filter_t *RandomInitFcFilter(uint32_t width, uint32_t height, uint32_t bits) {

  fc_filter_t *fc = (fc_filter_t *)malloc(sizeof(fc_filter_t));
  fc->width = width;
  fc->height = height;
  fc->order = 1;
  fc->bias = 0;

  if (bits == 8) {
    fc->vwidth = 0x8;
    fc->scale = 128;
  }
  else if (bits == 4) {
    fc->vwidth = 0x4;
    fc->scale = 8;
  }
  else if (bits == 2) {
    fc->vwidth = 0x2;
    fc->scale = 2;
  }
  else if (bits == 1) {
    fc->vwidth = 0x1;
    fc->scale = 1;
  }
  else {
    free(fc);
    return NULL;
  }

  int size = round_up_div(width * height * bits, 64);
  uint64_t *fc_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  uint16_t *data = (uint16_t *)fc_data;
  for (int i=0; i<size*4; i++) {
    // RAND_MAX assumed to be 32767
    data[i] = rand();
  }
  fc->addr = (void *)fc_data;

  return fc;
}

fc_filter_t *RandomInitFcFilterArray(uint32_t width, uint32_t height, uint32_t bits, int units) {

  fc_filter_t *fc = (fc_filter_t *)malloc(sizeof(fc_filter_t) * units);

  for (int i=0; i<units; i++) {
    fc[i] = *RandomInitFcFilter(width, height, bits);
  }

  return fc;
}

// read from array
image_t *InitImage_SC(uint32_t width, uint32_t height, uint32_t bits, uint8_t order, uint16_t scale, uint16_t zero_point, void *src) {

  image_t *img = (image_t *)malloc(sizeof(image_t));
  img->width = width;
  img->height = height;
  img->order = order;
  img->scale = scale;
  img->zero_point = zero_point;

  if (bits == 16) {
    img->vwidth = 0x80;
  }
  else if (bits == 8) {
    img->vwidth = 0x40;
  }
  else if (bits == 4) {
    img->vwidth = 0x20;
  }
  else if (bits == 2) {
    img->vwidth = 0x10;
  }
  else {
    free(img);
    return NULL;
  }

  int size = round_up_div(width * height * bits, 64);
  uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  memcpy(img_data, src, round_up_div(width * height * bits, 8));
  img->addr = (void *)img_data;

  return img;
}

image_mc_t *InitImage(uint32_t width, uint32_t height, uint32_t bits, uint16_t channel, uint8_t order, uint16_t scale, uint16_t zero_point, void *src) {
  
  image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
  img_mc->width = width;
  img_mc->height = height;
  img_mc->channel = channel;
  img_mc->order = 1;

  if (! ((bits == 16) || (bits == 8) || (bits == 4) || (bits == 2)) ) {
    free(img_mc);
    return NULL;
  }

  for (int i=0; i<channel; i++) {
    image_t *img = InitImage_SC(width, height, bits, order, scale, zero_point, (char *)src + round_up_div(width * height * bits, 8) * i);
    if (order == 1) {
      img_mc->img[i] = img;
    }
    else {
      image_t *img2 = Transpose(img);
      img_mc->img[i] = img2;
    }
  }

  return img_mc;
}

kernel_t *InitKernel_SC(uint32_t k, uint32_t bits, uint16_t scale, int bias, void *src) {

  kernel_t *kernel = (kernel_t *)malloc(sizeof(kernel_t));
  kernel->size = k;
  kernel->scale = scale;
  kernel->bias = bias;

  if (bits == 8) {
    kernel->vwidth = 0x8;
  }
  else if (bits == 4) {
    kernel->vwidth = 0x4;
  }
  else if (bits == 2) {
    kernel->vwidth = 0x2;
  }
  else if (bits == 1) {
    kernel->vwidth = 0x1;
  }
  else {
    free(kernel);
    return NULL;
  }

  int size = round_up_div(k * k * bits, 64);
  uint64_t *kernel_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  memcpy(kernel_data, src, round_up_div(k * k * bits, 8));
  kernel->addr = (void *)kernel_data;

  return kernel;
}

kernel_mc_t *InitKernel(uint32_t k, uint32_t bits, uint16_t in_channel, uint16_t out_channel, uint16_t scale, int *bias, void *src) {

  kernel_mc_t *ker_mc = (kernel_mc_t *)malloc(sizeof(kernel_mc_t));
  ker_mc->size = k;
  ker_mc->in_channel = in_channel;
  ker_mc->out_channel = out_channel;

  if (! ((bits == 8) || (bits == 4) || (bits == 2) || (bits == 1)) ) {
    free(ker_mc);
    return NULL;
  }

  for (int i=0; i<in_channel*out_channel; i++) {
    ker_mc->ker[i] = InitKernel_SC(k, bits, scale, bias[i / in_channel], (char *)src + round_up_div(k * k * bits, 8) * i);
  }

  return ker_mc;
}

fc_filter_t *InitFcFilter(uint32_t width, uint32_t height, uint32_t bits, uint16_t scale, int bias, void *src) {

  assert(width == 1);

  fc_filter_t *fc = (fc_filter_t *)malloc(sizeof(fc_filter_t));
  fc->width = width;
  fc->height = height;
  fc->order = 1;
  fc->scale = scale;
  fc->bias = bias;

  if (bits == 8) {
    fc->vwidth = 0x8;
  }
  else if (bits == 4) {
    fc->vwidth = 0x4;
  }
  else if (bits == 2) {
    fc->vwidth = 0x2;
  }
  else if (bits == 1) {
    fc->vwidth = 0x1;
  }
  else {
    free(fc);
    return NULL;
  }

  int size = round_up_div(width * height * bits, 64);
  uint64_t *fc_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  memcpy(fc_data, src, round_up_div(width * height * bits, 8));
  fc->addr = (void *)fc_data;

  return fc;
}

fc_filter_t *InitFcFilterArray(uint32_t width, uint32_t height, uint32_t bits, int units, uint16_t scale, int *bias, void *src) {

  fc_filter_t *fc = (fc_filter_t *)malloc(sizeof(fc_filter_t) * units);

  for (int i=0; i<units; i++) {
    fc[i] = *InitFcFilter(width, height, bits, scale, bias[i], (char *)src + round_up_div(width * height * bits, 8) * i);
  }

  return fc;
}

// output
void SetOutput_SC(image_t *output_image) {

  int width = output_image->width;
  int height = output_image->height;
  uint8_t vwidth = output_image->vwidth;
  uint64_t *img_addr = output_image->addr;

  printf("width: %d, height: %d, vwidth: %#x, order: %d, scale: %d, zero: %d\n", width, height, vwidth, output_image->order, output_image->scale, output_image->zero_point);

  if (output_image->order == 0) {
    for (int i=0; i<height; i++) {
      for (int j=0; j<width; j++) {
        printf("  %d", get_main_value(img_addr, i * width + j, vwidth));
      }
      printf("\n");
    }
  }
  else {
    for (int i=0; i<height; i++) {
      for (int j=0; j<width; j++) {
        printf("  %d", get_main_value(img_addr, j * height + i, vwidth));
      }
      printf("\n");
    }
  }
}

void SetOutputKernel_SC(kernel_t *output_kernel) {

  int k = output_kernel->size;
  uint8_t vwidth = output_kernel->vwidth;
  uint64_t *kernel_addr = output_kernel->addr;

  printf("k: %d, vwidth: %#x, scale: %d, bias: %d\n", k, vwidth, output_kernel->scale, output_kernel->bias);

  for (int i=0; i<k; i++) {
    for (int j=0; j<k; j++) {
      printf("  %d", get_kernel_value(kernel_addr, i * k + j, vwidth));
    }
    printf("\n");
  }
}

void SetOutput(image_mc_t *output_image) {

  printf("\nwidth: %d, height: %d, channel: %d\n", output_image->width, output_image->height, output_image->channel);

  for (int i=0; i<output_image->channel; i++) {
    printf("channel #%d: \n", i);
    SetOutput_SC(output_image->img[i]);
  }
}

void SetOutputKernel(kernel_mc_t *output_kernel) {

  printf("\nk: %d, channel_out: %d, channel_in: %d\n", output_kernel->size, output_kernel->out_channel, output_kernel->in_channel);

  for (int i=0; i<output_kernel->out_channel; i++) {
    printf("out_channel #%d: \n", i);
    for (int j=0; j<output_kernel->in_channel; j++) {
      printf("in_channel #%d: \n", j);
      SetOutputKernel_SC(output_kernel->ker[i*output_kernel->in_channel+j]);
    }
  }
}

void SetOutputFcFilter(fc_filter_t *output_fc_filter) {

  int width = output_fc_filter->width;
  int height = output_fc_filter->height;
  uint8_t vwidth = output_fc_filter->vwidth;
  uint64_t *img_addr = output_fc_filter->addr;

  printf("\nwidth: %d, height: %d, vwidth: %#x, order: %d, scale: %d, bias: %d\n", width, height, vwidth, output_fc_filter->order, output_fc_filter->scale, output_fc_filter->bias);

  if (output_fc_filter->order == 0) {
    for (int i=0; i<height; i++) {
      for (int j=0; j<width; j++) {
        printf("  %d", get_kernel_value(img_addr, i * width + j, vwidth));
      }
      printf("\n");
    }
  }
  else {
    for (int i=0; i<height; i++) {
      for (int j=0; j<width; j++) {
        printf("  %d", get_kernel_value(img_addr, j * height + i, vwidth));
      }
      printf("\n");
    }
  }
}

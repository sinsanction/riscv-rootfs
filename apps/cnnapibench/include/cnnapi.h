#ifndef __CNNAPI_H__
#define __CNNAPI_H__

#ifdef __nutshell_am
#include <am.h>
#include <klib.h>
#include <klib-macros.h>
#else
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHANNEL 200

typedef struct image
{
  uint32_t width;
  uint32_t height;

  uint8_t vwidth;
  uint8_t order;         // 0: row  1: column

  uint16_t scale;
  uint16_t zero_point;

  void *addr;
} image_t;

typedef struct kernel
{
  uint32_t size;
  uint8_t vwidth;

  uint16_t scale;
  int bias;

  void *addr;
} kernel_t;

typedef struct fc_filter
{
  uint32_t width;
  uint32_t height;

  uint8_t vwidth;
  uint8_t order;
  uint16_t scale;
  int bias;

  void *addr;
} fc_filter_t;

typedef struct image_mc
{
  uint32_t width;
  uint32_t height;
  uint16_t channel;

  uint8_t order;

  image_t *img[MAX_CHANNEL];
} image_mc_t;

typedef struct kernel_mc
{
  uint32_t size;
  uint16_t in_channel;
  uint16_t out_channel;

  kernel_t *ker[MAX_CHANNEL];
} kernel_mc_t;

typedef struct out_scale
{
  uint16_t scale;
  uint16_t zero_point;
} out_scale_t;

typedef struct out_scale_mc
{
  uint16_t channel;
  out_scale_t *scale;
} out_scale_mc_t;

//IO
image_t *RandomInitImage_SC(uint32_t width, uint32_t height, uint32_t bits, uint8_t order);

kernel_t *RandomInitKernel_SC(uint32_t k, uint32_t bits);

image_mc_t *RandomInitImage(uint32_t width, uint32_t height, uint32_t bits, uint16_t channel);

kernel_mc_t *RandomInitKernel(uint32_t k, uint32_t bits, uint16_t in_channel, uint16_t out_channel);

fc_filter_t *RandomInitFcFilter(uint32_t width, uint32_t height, uint32_t bits);

fc_filter_t *RandomInitFcFilterArray(uint32_t width, uint32_t height, uint32_t bits, int units);

image_t *InitImage_SC(uint32_t width, uint32_t height, uint32_t bits, uint8_t order, uint16_t scale, uint16_t zero_point, void *src);

image_mc_t *InitImage(uint32_t width, uint32_t height, uint32_t bits, uint16_t channel, uint8_t order, uint16_t scale, uint16_t zero_point, void *src);

kernel_t *InitKernel_SC(uint32_t k, uint32_t bits, uint16_t scale, int bias, void *src);

kernel_mc_t *InitKernel(uint32_t k, uint32_t bits, uint16_t in_channel, uint16_t out_channel, uint16_t scale, int *bias, void *src);

fc_filter_t *InitFcFilter(uint32_t width, uint32_t height, uint32_t bits, uint16_t scale, int bias, void *src);

fc_filter_t *InitFcFilterArray(uint32_t width, uint32_t height, uint32_t bits, int units, uint16_t scale, int *bias, void *src);

void SetOutput_SC(image_t *output_image);

void SetOutputKernel_SC(kernel_t *output_kernel);

void SetOutput(image_mc_t *output_image);

void SetOutputKernel(kernel_mc_t *output_kernel);

void SetOutputFcFilter(fc_filter_t *output_fc_filter);

//utils
image_t *Transpose(image_t *input_image);

image_t *MergeImage(image_t *input_image_a, image_t *input_image_b);

void Rescale_SC(image_t *input_image, out_scale_t *out_scale);

void Rescale(image_mc_t *input_image, out_scale_mc_t *out_scale);

//arithmetic
image_t *Convolution_SC(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale);

image_t *MaxPooling_SC(image_t *input_image, int pool_size, int strides);

image_t *AvgPooling_SC(image_t *input_image, int pool_size, int strides);

image_t *Activation_SC(image_t *input_image, char *algorithm, uint16_t zero_point);


image_mc_t *Convolution(image_mc_t *input_image, kernel_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale);

image_mc_t *MaxPooling(image_mc_t *input_image, int pool_size, int strides);

image_mc_t *AvgPooling(image_mc_t *input_image, int pool_size, int strides);

image_mc_t *Activation(image_mc_t *input_image, char *algorithm, uint16_t zero_point);

image_t *Flatten(image_mc_t *input_image);

image_t *Dense(image_t *input_image, fc_filter_t *fc_filter_array, int units, out_scale_t *out_scale);

#ifdef __cplusplus
}
#endif

#endif

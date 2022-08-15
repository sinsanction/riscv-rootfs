#ifndef __CNNAPI_V2_H__
#define __CNNAPI_V2_H__

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

// full mixed-precision

typedef struct image_mp
{
  uint32_t width;
  uint32_t height;

  uint8_t *vwidth;

  uint16_t scale;
  uint16_t zero_point;

  void **addr;
} image_mp_t;

typedef struct kernel_mp
{
  uint32_t size;
  uint8_t *vwidth;

  uint16_t scale;

  void **addr;
} kernel_mp_t;

typedef struct fc_filter_mp
{
  uint32_t width;
  uint32_t height;

  uint8_t *vwidth;
  uint16_t scale;

  void **addr;
} fc_filter_mp_t;

typedef struct image_mp_mc
{
  uint32_t width;
  uint32_t height;
  uint16_t channel;

  image_mp_t *img[MAX_CHANNEL];
} image_mp_mc_t;

typedef struct kernel_mp_mc
{
  uint32_t size;
  uint16_t in_channel;
  uint16_t out_channel;

  kernel_mp_t *ker[MAX_CHANNEL];
} kernel_mp_mc_t;

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
image_mp_t *RandomInitImage_MP_SC(uint32_t width, uint32_t height);

kernel_mp_t *RandomInitKernel_MP_SC(uint32_t k);

image_mp_mc_t *RandomInitImage_MP(uint32_t width, uint32_t height, uint16_t channel);

kernel_mp_mc_t *RandomInitKernel_MP(uint32_t k, uint16_t in_channel, uint16_t out_channel);

fc_filter_mp_t *RandomInitFcFilter_MP(uint32_t width, uint32_t height);

fc_filter_mp_t *RandomInitFcFilterArray_MP(uint32_t width, uint32_t height, int units);

void SetOutput_MP_SC(image_mp_t *output_image);

void SetOutputKernel_MP_SC(kernel_mp_t *output_kernel);

void SetOutput_MP(image_mp_mc_t *output_image);

void SetOutputKernel_MP(kernel_mp_mc_t *output_kernel);

void SetOutputFcFilter_MP(fc_filter_mp_t *output_fc_filter);

//arithmetic
image_mp_t *Convolution_MP_SC(image_mp_t *input_image, kernel_mp_t *input_kernel, int strides, out_scale_t *out_scale);

image_mp_t *MaxPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides);

image_mp_t *AvgPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides);

image_mp_t *Activation_MP_SC(image_mp_t *input_image, char *algorithm, uint16_t *zero_point);


image_mp_mc_t *Convolution_MP(image_mp_mc_t *input_image, kernel_mp_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale);

image_mp_mc_t *MaxPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides);

image_mp_mc_t *AvgPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides);

image_mp_mc_t *Activation_MP(image_mp_mc_t *input_image, char *algorithm, uint16_t *zero_point);

image_mp_t *Flatten_MP(image_mp_mc_t *input_image);

image_mp_t *Dense_MP(image_mp_t *input_image, fc_filter_mp_t *fc_filter_array, int units, out_scale_t *out_scale);

#ifdef __cplusplus
}
#endif

#endif

#ifndef __CNNAPI_STDINS_H__
#define __CNNAPI_STDINS_H__

#include "cnnapi.h"

#ifdef __cplusplus
extern "C" {
#endif

image_t *StdIns_Convolution_SC(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale);

image_t *StdIns_MaxPooling_SC(image_t *input_image, int pool_size, int strides);

image_t *StdIns_AvgPooling_SC(image_t *input_image, int pool_size, int strides);

image_t *StdIns_Activation_SC(image_t *input_image, char *algorithm, uint16_t zero_point);


image_mc_t *StdIns_Convolution(image_mc_t *input_image, kernel_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale);

image_mc_t *StdIns_MaxPooling(image_mc_t *input_image, int pool_size, int strides);

image_mc_t *StdIns_AvgPooling(image_mc_t *input_image, int pool_size, int strides);

image_mc_t *StdIns_Activation(image_mc_t *input_image, char *algorithm, uint16_t zero_point);

#ifdef __cplusplus
}
#endif

#endif

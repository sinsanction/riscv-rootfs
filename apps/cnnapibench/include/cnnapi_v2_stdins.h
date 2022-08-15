#ifndef __CNNAPI_V2_STDINS_H__
#define __CNNAPI_V2_STDINS_H__

#include "cnnapi_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

image_mp_t *StdIns_Convolution_MP_SC(image_mp_t *input_image, kernel_mp_t *input_kernel, int strides, out_scale_t *out_scale);

image_mp_t *StdIns_MaxPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides);

image_mp_t *StdIns_AvgPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides);

image_mp_t *StdIns_Activation_MP_SC(image_mp_t *input_image, char *algorithm, uint16_t *zero_point);


image_mp_mc_t *StdIns_Convolution_MP(image_mp_mc_t *input_image, kernel_mp_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale);

image_mp_mc_t *StdIns_MaxPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides);

image_mp_mc_t *StdIns_AvgPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides);

image_mp_mc_t *StdIns_Activation_MP(image_mp_mc_t *input_image, char *algorithm, uint16_t *zero_point);

#ifdef __cplusplus
}
#endif

#endif

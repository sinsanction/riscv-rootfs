#include "cnnapi_v2.h"
#include "cnnapi_common.h"


//conv
image_mp_t *StdIns_Convolution_MP_SC(image_mp_t *input_image, kernel_mp_t *input_kernel, int strides, out_scale_t *out_scale) {

    assert((input_kernel->size <= input_image->width) && (input_kernel->size <= input_image->height));
    assert((input_kernel->size <= 5) && (input_kernel->size >= 1));
    assert(input_kernel->scale != 0);

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = (input_image->width - input_kernel->size) / strides + 1;
    img->height = (input_image->height - input_kernel->size) / strides + 1;
    int vwidth_size = round_up_div(img->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * img->width);
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    for (int i=0; i<img->width; i++) {
        uint8_t vwidth_max = input_image->vwidth[i * strides];
        for (int j=1; j<input_kernel->size; j++) {
            vwidth_max = (vwidth_max >= input_image->vwidth[i * strides + j]) ? vwidth_max : input_image->vwidth[i * strides + j];
        }
        img->vwidth[i] = vwidth_max;
        int size = round_up_div(img->height * (img->vwidth[i] >> 3), 64);
        uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
        img->addr[i] = (void *)img_data;
    }

    int width = img->width;
    int height = img->height;
    int k = input_kernel->size;
    int temp;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int si=0; si<k; si++) {
                for (int sj=0; sj<k; sj++) {
                    temp += get_main_value((uint64_t *)(input_image->addr[j * strides + sj]), i * strides + si, input_image->vwidth[j * strides + sj]) * get_kernel_value((uint64_t *)(input_kernel->addr[si]), sj, input_kernel->vwidth[si]);
                }
            }
            temp = (temp < 0) ? 0 : temp;
            temp = temp / input_kernel->scale;
            temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
            put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], handle_overflow(temp, img->vwidth[j]));
        }
    }

    return img;
}

//max pool
image_mp_t *StdIns_MaxPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = (input_image->width - pool_size) / strides + 1;
    img->height = (input_image->height - pool_size) / strides + 1;
    int vwidth_size = round_up_div(img->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * img->width);
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    for (int i=0; i<img->width; i++) {
        uint8_t vwidth_max = input_image->vwidth[i * strides];
        for (int j=1; j<pool_size; j++) {
            vwidth_max = (vwidth_max >= input_image->vwidth[i * strides + j]) ? vwidth_max : input_image->vwidth[i * strides + j];
        }
        img->vwidth[i] = vwidth_max;
        int size = round_up_div(img->height * (img->vwidth[i] >> 3), 64);
        uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
        img->addr[i] = (void *)img_data;
    }

    int width = img->width;
    int height = img->height;
    int k = pool_size;
    uint16_t temp;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int sj=0; sj<k; sj++) {
                for (int si=0; si<k; si++) {
                    uint16_t new_data = get_main_value((uint64_t *)(input_image->addr[j * strides + sj]), i * strides + si, input_image->vwidth[j * strides + sj]);
                    temp = (temp > new_data) ? temp : new_data;
                }
            }
            put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], temp);
        }
    }

    return img;
}

//avg pool
image_mp_t *StdIns_AvgPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = (input_image->width - pool_size) / strides + 1;
    img->height = (input_image->height - pool_size) / strides + 1;
    int vwidth_size = round_up_div(img->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * img->width);
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    for (int i=0; i<img->width; i++) {
        uint8_t vwidth_max = input_image->vwidth[i * strides];
        for (int j=1; j<pool_size; j++) {
            vwidth_max = (vwidth_max >= input_image->vwidth[i * strides + j]) ? vwidth_max : input_image->vwidth[i * strides + j];
        }
        img->vwidth[i] = vwidth_max;
        int size = round_up_div(img->height * (img->vwidth[i] >> 3), 64);
        uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
        img->addr[i] = (void *)img_data;
    }

    int width = img->width;
    int height = img->height;
    int k = pool_size;
    uint32_t temp;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int sj=0; sj<k; sj++) {
                for (int si=0; si<k; si++) {
                    temp += get_main_value((uint64_t *)(input_image->addr[j * strides + sj]), i * strides + si, input_image->vwidth[j * strides + sj]);
                }
            }
            int div = temp / (k * k);
            int rem = temp % (k * k);
            int cin = ((rem * 2) >= (k * k)) ? 1 : 0;
            put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], div + cin);
        }
    }

    return img;
}

//act
image_mp_t *StdIns_Activation_MP_SC(image_mp_t *input_image, char *algorithm, uint16_t *zero_point) {

    assert(strcmp(algorithm, "relu") == 0);

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = input_image->width;
    img->height = input_image->height;
    int vwidth_size = round_up_div(input_image->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * input_image->width);
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    for (int i=0; i<img->width; i++) {
        img->vwidth[i] = input_image->vwidth[i];
        int size = round_up_div(img->height * (img->vwidth[i] >> 3), 64);
        uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
        uint64_t *inimg_data = (uint64_t *)(input_image->addr[i]);

        for (int j=0; j<img->height; j++) {
            uint16_t new_data = get_main_value(inimg_data, j, input_image->vwidth[i]);
            uint16_t res = (new_data > zero_point[i]) ? new_data : zero_point[i];
            put_main_value(img_data, j, img->vwidth[i], res);
        }
        img->addr[i] = (void *)img_data;
    }

    return img;
}

//multi channel
image_mp_mc_t *StdIns_Convolution_MP(image_mp_mc_t *input_image, kernel_mp_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale) {

    assert(input_image->channel == input_kernel->in_channel);
    assert(input_kernel->out_channel == out_scale->channel);

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = (input_image->width - input_kernel->size) / strides + 1;
    img_mc->height = (input_image->height - input_kernel->size) / strides + 1;
    img_mc->channel = input_kernel->out_channel;

    image_mp_t **img_tmp;
    kernel_mp_t *curr_kernel;
    uint32_t temp;
    img_tmp = (image_mp_t **)malloc(sizeof(image_mp_t *) * input_image->channel);

    for (int i=0; i<img_mc->channel; i++) {
        for (int j=0; j<input_image->channel; j++) {
            curr_kernel = input_kernel->ker[i*input_kernel->in_channel+j];
            img_tmp[j] = StdIns_Convolution_MP_SC(input_image->img[j], curr_kernel, strides, &(out_scale->scale[i]));
        }

        //merge all channel
        image_mp_t *new_img = (image_mp_t *)malloc(sizeof(image_mp_t));
        new_img->width = img_mc->width;
        new_img->height = img_mc->height;
        int vwidth_size = round_up_div(new_img->width, 8);
        new_img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
        new_img->addr = (void **)malloc(sizeof(void *) * new_img->width);
        new_img->scale = out_scale->scale[i].scale;
        new_img->zero_point = out_scale->scale[i].zero_point;

        for (int j=0; j<new_img->width; j++) {
            uint8_t vwidth_max = img_tmp[0]->vwidth[j];
            for (int l=1; l<input_image->channel; l++) {
                vwidth_max = (vwidth_max >= img_tmp[l]->vwidth[j]) ? vwidth_max : img_tmp[l]->vwidth[j];
            }
            new_img->vwidth[j] = vwidth_max;
            int size = round_up_div(new_img->height * (new_img->vwidth[j] >> 3), 64);
            uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
            new_img->addr[j] = (void *)img_data;
        }

        for (int j=0; j<new_img->width; j++) {
            for (int i=0; i<new_img->height; i++) {
                temp = 0;
                for (int l=0; l<input_image->channel; l++) {
                    temp += (get_main_value((uint64_t *)(img_tmp[l]->addr[j]), i, img_tmp[l]->vwidth[j]) - img_tmp[l]->zero_point);
                }
                temp = temp + new_img->zero_point;
                temp = handle_overflow(temp, new_img->vwidth[j]);
                put_main_value((uint64_t *)(new_img->addr[j]), i, new_img->vwidth[j], temp);
            }
        }

        img_mc->img[i] = new_img;

        for (int j=0; j<input_image->channel; j++) {
            free(img_tmp[j]);
        }
    }

    free(img_tmp);
    return img_mc;
}

image_mp_mc_t *StdIns_MaxPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_MaxPooling_MP_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mp_mc_t *StdIns_AvgPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_AvgPooling_MP_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mp_mc_t *StdIns_Activation_MP(image_mp_mc_t *input_image, char *algorithm, uint16_t *zero_point) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = input_image->width;
    img_mc->height = input_image->height;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_Activation_MP_SC(input_image->img[i], algorithm, &zero_point[i * input_image->width]);
    }

    return img_mc;
}


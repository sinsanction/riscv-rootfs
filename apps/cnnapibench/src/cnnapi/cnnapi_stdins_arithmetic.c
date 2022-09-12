#include "cnnapi.h"
#include "cnnapi_common.h"


//conv
image_t *StdIns_Convolution_SC(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {

    assert((input_kernel->size <= input_image->width) && (input_kernel->size <= input_image->height));
    assert((input_kernel->size <= 5) && (input_kernel->size >= 1));
    assert(input_kernel->scale != 0);
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));
    assert((input_kernel->vwidth == 0x8) || (input_kernel->vwidth == 0x4) || (input_kernel->vwidth == 0x2) || (input_kernel->vwidth == 0x1));

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - input_kernel->size) / strides + 1;
    img->height = (input_image->height - input_kernel->size) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    int k = input_kernel->size;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    int temp;

    int ker_sum = 0;
    for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
            ker_sum += get_kernel_value(inker_data, si * k + sj, vwidth_kernel);
        }
    }

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int si=0; si<k; si++) {
                for (int sj=0; sj<k; sj++) {
                    temp += get_main_value(inimg_data, (j * strides + sj) * input_image->height + (i * strides + si), vwidth_main) * get_kernel_value(inker_data, si * k + sj, vwidth_kernel);
                }
            }
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *StdIns_Convolution_SC_Inter(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {

    assert((input_kernel->size <= input_image->width) && (input_kernel->size <= input_image->height));
    assert((input_kernel->size <= 5) && (input_kernel->size >= 1));
    assert(input_kernel->scale != 0);
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));
    assert((input_kernel->vwidth == 0x8) || (input_kernel->vwidth == 0x4) || (input_kernel->vwidth == 0x2) || (input_kernel->vwidth == 0x1));

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - input_kernel->size) / strides + 1;
    img->height = (input_image->height - input_kernel->size) / strides + 1;
    img->vwidth = 0x80; //16bit
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    int k = input_kernel->size;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (img->vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    int temp;

    int ker_sum = 0;
    for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
            ker_sum += get_kernel_value(inker_data, si * k + sj, vwidth_kernel);
        }
    }

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int si=0; si<k; si++) {
                for (int sj=0; sj<k; sj++) {
                    temp += get_main_value(inimg_data, (j * strides + sj) * input_image->height + (i * strides + si), vwidth_main) * get_kernel_value(inker_data, si * k + sj, vwidth_kernel);
                }
            }
            temp = temp - input_image->zero_point * ker_sum;
            temp = (temp > 32767) ? 32767 : ((temp < -32767) ? -32767 : temp);
            put_main_value(img_data, j * height + i, img->vwidth, temp);
        }
    }

    img->addr = (void *)img_data;
    return img;
}

//max pool
image_t *StdIns_MaxPooling_SC(image_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - pool_size) / strides + 1;
    img->height = (input_image->height - pool_size) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    int k = pool_size;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int sj=0; sj<k; sj++) {
                for (int si=0; si<k; si++) {
                    uint16_t new_data = get_main_value(inimg_data, (j * strides + sj) * input_image->height + (i * strides + si), vwidth);
                    temp = (temp > new_data) ? temp : new_data;
                }
            }
            put_main_value(img_data, j * height + i, vwidth, temp);
        }
    }

    img->addr = (void *)img_data;
    return img;
}

//avg pool
image_t *StdIns_AvgPooling_SC(image_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - pool_size) / strides + 1;
    img->height = (input_image->height - pool_size) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    int k = pool_size;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint32_t temp;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            temp = 0;
            for (int sj=0; sj<k; sj++) {
                for (int si=0; si<k; si++) {
                    temp += get_main_value(inimg_data, (j * strides + sj) * input_image->height + (i * strides + si), vwidth);
                }
            }
            int div = temp / (k * k);
            int rem = temp % (k * k);
            int cin = ((rem * 2) >= (k * k)) ? 1 : 0;
            put_main_value(img_data, j * height + i, vwidth, div + cin);
        }
    }

    img->addr = (void *)img_data;
    return img;
}

//act
image_t *StdIns_Activation_SC(image_t *input_image, char *algorithm, uint16_t zero_point) {

    assert(strcmp(algorithm, "relu") == 0);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = input_image->width;
    img->height = input_image->height;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = input_image->width;
    int height = input_image->height;
    uint8_t vwidth = input_image->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);

    for (int i=0; i<width*height; i++) {
        uint16_t new_data = get_main_value(inimg_data, i, vwidth);
        uint16_t res = (new_data > zero_point) ? new_data : zero_point;
        put_main_value(img_data, i, vwidth, res);
    }

    img->addr = (void *)img_data;
    return img;
}

//multi channel
image_mc_t *StdIns_Convolution(image_mc_t *input_image, kernel_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale) {

    assert(input_image->channel == input_kernel->in_channel);
    assert(input_kernel->out_channel == out_scale->channel);

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = (input_image->width - input_kernel->size) / strides + 1;
    img_mc->height = (input_image->height - input_kernel->size) / strides + 1;
    img_mc->channel = input_kernel->out_channel;
    img_mc->order = input_image->order;

    image_t **img_tmp;
    kernel_t *curr_kernel;
    int temp;
    uint8_t vwidth_max = 0;
    img_tmp = (image_t **)malloc(sizeof(image_t *) * input_image->channel);

    for (int i=0; i<input_image->channel; i++) {
        vwidth_max = (input_image->img[i]->vwidth > vwidth_max) ? input_image->img[i]->vwidth : vwidth_max;
    }

    for (int i=0; i<img_mc->channel; i++) {
        for (int j=0; j<input_image->channel; j++) {
            curr_kernel = input_kernel->ker[i*input_kernel->in_channel+j];
            img_tmp[j] = StdIns_Convolution_SC_Inter(input_image->img[j], curr_kernel, strides, &(out_scale->scale[i]));
        }

        //merge all channel
        image_t *new_img = (image_t *)malloc(sizeof(image_t));
        new_img->width = img_mc->width;
        new_img->height = img_mc->height;
        new_img->vwidth = vwidth_max;
        new_img->order = 1;
        new_img->scale = out_scale->scale[i].scale;
        new_img->zero_point = out_scale->scale[i].zero_point;

        int size = round_up_div(new_img->width * new_img->height * (vwidth_max >> 3), 64);
        uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);

        for (int j=0; j<new_img->width; j++) {
            for (int i=0; i<new_img->height; i++) {
                temp = 0;
                for (int l=0; l<input_image->channel; l++) {
                    temp += (int16_t)get_main_value((uint64_t *)(img_tmp[l]->addr), j * new_img->height + i, img_tmp[l]->vwidth);
                }
                temp = temp + curr_kernel->bias;
                temp = temp * new_img->scale / (input_image->img[0]->scale * curr_kernel->scale) + new_img->zero_point;
                temp = handle_overflow(temp, vwidth_max);
                put_main_value(img_data, j * new_img->height + i, vwidth_max, temp);
            }
        }

        new_img->addr = (void *)img_data;
        img_mc->img[i] = new_img;

        for (int i=0; i<input_image->channel; i++) {
            free(img_tmp[i]);
        }
    }

    free(img_tmp);
    return img_mc;
}

image_mc_t *StdIns_MaxPooling(image_mc_t *input_image, int pool_size, int strides) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_MaxPooling_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mc_t *StdIns_AvgPooling(image_mc_t *input_image, int pool_size, int strides) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_AvgPooling_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mc_t *StdIns_Activation(image_mc_t *input_image, char *algorithm, uint16_t zero_point) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = input_image->width;
    img_mc->height = input_image->height;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = StdIns_Activation_SC(input_image->img[i], algorithm, zero_point);
    }

    return img_mc;
}


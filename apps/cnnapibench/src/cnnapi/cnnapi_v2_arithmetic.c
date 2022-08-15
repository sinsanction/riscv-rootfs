#include "cnninst.h"
#include "cnnapi_v2.h"
#include "cnnapi_common.h"


//conv
image_mp_t *Convolution_MP_SC(image_mp_t *input_image, kernel_mp_t *input_kernel, int strides, out_scale_t *out_scale) {

    assert((input_kernel->size <= input_image->width) && (input_kernel->size <= input_image->height));
    assert((input_kernel->size <= 5) && (input_kernel->size >= 1));

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
    uint32_t temp;
    uint64_t col_ptr;

    uint64_t *vwidth_reg = (uint64_t *)(input_kernel->vwidth);
    LoadV_Width((uint64_t)vwidth_reg);
    for (int i=0; i<input_kernel->size; i++) {
        uint64_t kernel_ptr = get_addr64_kernel((uint64_t *)(input_kernel->addr[i]), 0, input_kernel->vwidth[i]);
        LoadV_D_Kernel(kernel_ptr, input_kernel->size, i, i);
    }

    vwidth_reg = (uint64_t *)(input_image->vwidth);
    int count = 0;
    LoadV_Width((uint64_t)vwidth_reg);

    if (strides >= input_kernel->size) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                for (int l=0; l<input_kernel->size; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[j * strides + l]), i * strides, input_image->vwidth[j * strides + l]);
                    LoadV_D_Main(col_ptr, input_kernel->size, l, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Conv(input_kernel->size);
                temp = temp / input_kernel->scale;
                temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], handle_overflow(temp, img->vwidth[j]));
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            for (int l=0; l<input_kernel->size; l++) {
                col_ptr = get_addr64((uint64_t *)(input_image->addr[l]), i * strides, input_image->vwidth[l]);
                LoadV_D_Main(col_ptr, input_kernel->size, l, count);
                count++;
                if (count >= 8) {
                    count = 0;
                    vwidth_reg++;
                    LoadV_Width((uint64_t)vwidth_reg);
                }
            }
            temp = Conv(input_kernel->size);
            temp = temp / input_kernel->scale;
            temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
            put_main_value((uint64_t *)(img->addr[0]), i, img->vwidth[0], handle_overflow(temp, img->vwidth[0]));

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[(j-1) * strides + input_kernel->size + l]), i * strides, input_image->vwidth[(j-1) * strides + input_kernel->size + l]);
                    LoadV_P(col_ptr, input_kernel->size, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Conv(input_kernel->size);
                temp = temp / input_kernel->scale;
                temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], handle_overflow(temp, img->vwidth[j]));
            }
        }
    }

    return img;
}

//max pool
image_mp_t *MaxPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides) {

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
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t *vwidth_reg = (uint64_t *)(input_image->vwidth);
    int count = 0;
    LoadV_Width((uint64_t)vwidth_reg);

    if (strides >= pool_size) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                for (int l=0; l<pool_size; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[j * strides + l]), i * strides, input_image->vwidth[j * strides + l]);
                    LoadV_D_Main(col_ptr, pool_size, l, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Pool_Max(pool_size);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            for (int l=0; l<pool_size; l++) {
                col_ptr = get_addr64((uint64_t *)(input_image->addr[l]), i * strides, input_image->vwidth[l]);
                LoadV_D_Main(col_ptr, pool_size, l, count);
                count++;
                if (count >= 8) {
                    count = 0;
                    vwidth_reg++;
                    LoadV_Width((uint64_t)vwidth_reg);
                }
            }
            temp = Pool_Max(pool_size);
            put_main_value((uint64_t *)(img->addr[0]), i, img->vwidth[0], temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[(j-1) * strides + pool_size + l]), i * strides, input_image->vwidth[(j-1) * strides + pool_size + l]);
                    LoadV_P(col_ptr, pool_size, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Pool_Max(pool_size);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], temp);
            }
        }
    }

    return img;
}

//avg pool
image_mp_t *AvgPooling_MP_SC(image_mp_t *input_image, int pool_size, int strides) {

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
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t *vwidth_reg = (uint64_t *)(input_image->vwidth);
    int count = 0;
    LoadV_Width((uint64_t)vwidth_reg);

    if (strides >= pool_size) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                for (int l=0; l<pool_size; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[j * strides + l]), i * strides, input_image->vwidth[j * strides + l]);
                    LoadV_D_Main(col_ptr, pool_size, l, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Pool_Avg(pool_size);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            for (int l=0; l<pool_size; l++) {
                col_ptr = get_addr64((uint64_t *)(input_image->addr[l]), i * strides, input_image->vwidth[l]);
                LoadV_D_Main(col_ptr, pool_size, l, count);
                count++;
                if (count >= 8) {
                    count = 0;
                    vwidth_reg++;
                    LoadV_Width((uint64_t)vwidth_reg);
                }
            }
            temp = Pool_Avg(pool_size);
            put_main_value((uint64_t *)(img->addr[0]), i, img->vwidth[0], temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    col_ptr = get_addr64((uint64_t *)(input_image->addr[(j-1) * strides + pool_size + l]), i * strides, input_image->vwidth[(j-1) * strides + pool_size + l]);
                    LoadV_P(col_ptr, pool_size, count);
                    count++;
                    if (count >= 8) {
                        count = 0;
                        vwidth_reg++;
                        LoadV_Width((uint64_t)vwidth_reg);
                    }
                }
                temp = Pool_Avg(pool_size);
                put_main_value((uint64_t *)(img->addr[j]), i, img->vwidth[j], temp);
            }
        }
    }

    return img;
}

//act
image_mp_t *Activation_MP_SC(image_mp_t *input_image, char *algorithm, uint16_t *zero_point) {

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

        uint64_t zero = 0;
        if (img->vwidth[i] == 0x80) {
            for (int j=0; j<4; j++) {
                zero |= (uint64_t)zero_point[i] << (j * 16);
            }
        }
        else if (img->vwidth[i] == 0x40) {
            uint16_t zero_p = zero_point[i] & 0xff;
            for (int j=0; j<8; j++) {
                zero |= (uint64_t)zero_p << (j * 8);
            }
        }
        else if (img->vwidth[i] == 0x20) {
            uint16_t zero_p = zero_point[i] & 0xf;
            for (int j=0; j<16; j++) {
                zero |= (uint64_t)zero_p << (j * 4);
            }
        }
        else { //img->vwidth[i] == 0x10
            uint16_t zero_p = zero_point[i] & 0x3;
            for (int j=0; j<32; j++) {
                zero |= (uint64_t)zero_p << (j * 2);
            }
        }

        uint64_t vwidth_reg = img->vwidth[i];
        LoadV_Width((uint64_t)&vwidth_reg);

        for (int j=0; j<size; j++) {
            img_data[j] = Act(inimg_data[j], zero);
        }
        img->addr[i] = (void *)img_data;
    }

    return img;
}

//multi channel
image_mp_mc_t *Convolution_MP(image_mp_mc_t *input_image, kernel_mp_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale) {

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
            img_tmp[j] = Convolution_MP_SC(input_image->img[j], curr_kernel, strides, &(out_scale->scale[i]));
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

image_mp_mc_t *MaxPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = MaxPooling_MP_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mp_mc_t *AvgPooling_MP(image_mp_mc_t *input_image, int pool_size, int strides) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = AvgPooling_MP_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mp_mc_t *Activation_MP(image_mp_mc_t *input_image, char *algorithm, uint16_t *zero_point) {

    image_mp_mc_t *img_mc = (image_mp_mc_t *)malloc(sizeof(image_mp_mc_t));
    img_mc->width = input_image->width;
    img_mc->height = input_image->height;
    img_mc->channel = input_image->channel;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = Activation_MP_SC(input_image->img[i], algorithm, &zero_point[i * input_image->width]);
    }

    return img_mc;
}

//fully connected
image_mp_t *Flatten_MP(image_mp_mc_t *input_image) {

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = input_image->width * input_image->channel;
    img->height = input_image->height;
    int vwidth_size = round_up_div(img->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * img->width);
    img->scale = input_image->img[0]->scale;
    img->zero_point = input_image->img[0]->zero_point;

    for (int c=0; c<input_image->channel; c++) {
        for (int j=0; j<input_image->width; j++) {
            img->vwidth[c * input_image->width + j] = input_image->img[c]->vwidth[j];
            img->addr[c * input_image->width + j] = input_image->img[c]->addr[j];
        }
    }

    return img;
}

image_mp_t *Dense_MP(image_mp_t *input_image, fc_filter_mp_t *fc_filter_array, int units, out_scale_t *out_scale) {

    assert(input_image->width == fc_filter_array[0].width);
    assert(input_image->height == fc_filter_array[0].height);
    assert(fc_filter_array[0].scale != 0);

    image_mp_t *img = (image_mp_t *)malloc(sizeof(image_mp_t));
    img->width = 1;
    img->height = units;
    int vwidth_size = round_up_div(img->width, 8);
    img->vwidth = (uint8_t *)malloc(sizeof(uint64_t) * vwidth_size);
    img->addr = (void **)malloc(sizeof(void *) * img->width);
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    uint8_t vwidth_max = input_image->vwidth[0];
    for (int i=1; i<input_image->width; i++) {
        vwidth_max = (vwidth_max >= input_image->vwidth[i]) ? vwidth_max : input_image->vwidth[i];
    }
    img->vwidth[0] = vwidth_max;
    int size = round_up_div(img->height * (vwidth_max >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    img->addr[0] = (void *)img_data;

    int width = input_image->width;
    int height = input_image->height;
    int temp = 0;

    for (int u=0; u<units; u++) {
        temp = 0;
        for (int j=0; j<width; j++) {
            for (int i=0; i<height; i++) {
                temp += get_main_value((uint64_t *)(input_image->addr[j]), i, input_image->vwidth[j]) * get_kernel_value((uint64_t *)(fc_filter_array[u].addr[j]), i, fc_filter_array[u].vwidth[j]);
            }
        }

        temp = temp / fc_filter_array[u].scale;
        temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
        temp = handle_overflow(temp, img->vwidth[0]);
        put_main_value(img_data, u, img->vwidth[0], temp);
    }

    return img;
}


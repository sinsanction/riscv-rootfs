#include "cnninst.h"
#include "cnnapi.h"
#include "cnnapi_common.h"


//conv
image_t *convolution_k5(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 5) / strides + 1;
    img->height = (input_image->height - 5) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    LoadV_D_Kernel(kernel_ptr,    5, 0, 0);
    LoadV_D_Kernel(kernel_ptr+5,  5, 1, 0);
    LoadV_D_Kernel(kernel_ptr+10, 5, 2, 0);
    LoadV_D_Kernel(kernel_ptr+15, 5, 3, 0);
    LoadV_D_Kernel(kernel_ptr+20, 5, 4, 0);

    int ker_sum = 0;
    for (int si=0; si<5; si++) {
        for (int sj=0; sj<5; sj++) {
            ker_sum += get_kernel_value(inker_data, si * 5 + sj, vwidth_kernel);
        }
    }

    if (strides >= 5) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
                LoadV_D_Main(col_ptr, 5, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 5, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 5, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth_main), 5, 3, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth_main), 5, 4, 0);
                temp = Conv(5);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, 5, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 5, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 5, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth_main), 5, 3, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth_main), 5, 4, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*5, vwidth_main);
            temp = Conv(5);
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, i, vwidth_main, handle_overflow(temp, vwidth_main));

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 5, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth_main);
                }
                temp = Conv(5);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *convolution_k4(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 4) / strides + 1;
    img->height = (input_image->height - 4) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    LoadV_D_Kernel(kernel_ptr,    4, 0, 0);
    LoadV_D_Kernel(kernel_ptr+4,  4, 1, 0);
    LoadV_D_Kernel(kernel_ptr+8,  4, 2, 0);
    LoadV_D_Kernel(kernel_ptr+12, 4, 3, 0);

    int ker_sum = 0;
    for (int si=0; si<4; si++) {
        for (int sj=0; sj<4; sj++) {
            ker_sum += get_kernel_value(inker_data, si * 4 + sj, vwidth_kernel);
        }
    }

    if (strides >= 4) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
                LoadV_D_Main(col_ptr, 4, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 4, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 4, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth_main), 4, 3, 0);
                temp = Conv(4);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, 4, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 4, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 4, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth_main), 4, 3, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*4, vwidth_main);
            temp = Conv(4);
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, i, vwidth_main, handle_overflow(temp, vwidth_main));

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 4, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth_main);
                }
                temp = Conv(4);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *convolution_k3(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 3) / strides + 1;
    img->height = (input_image->height - 3) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    LoadV_D_Kernel(kernel_ptr,   3, 0, 0);
    LoadV_D_Kernel(kernel_ptr+3, 3, 1, 0);
    LoadV_D_Kernel(kernel_ptr+6, 3, 2, 0);

    int ker_sum = 0;
    for (int si=0; si<3; si++) {
        for (int sj=0; sj<3; sj++) {
            ker_sum += get_kernel_value(inker_data, si * 3 + sj, vwidth_kernel);
        }
    }

    if (strides >= 3) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
                LoadV_D_Main(col_ptr, 3, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 3, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 3, 2, 0);
                temp = Conv(3);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, 3, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 3, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth_main), 3, 2, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*3, vwidth_main);
            temp = Conv(3);
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, i, vwidth_main, handle_overflow(temp, vwidth_main));

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 3, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth_main);
                }
                temp = Conv(3);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *convolution_k2(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 2) / strides + 1;
    img->height = (input_image->height - 2) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    LoadV_D_Kernel(kernel_ptr,   2, 0, 0);
    LoadV_D_Kernel(kernel_ptr+2, 2, 1, 0);

    int ker_sum = 0;
    for (int si=0; si<2; si++) {
        for (int sj=0; sj<2; sj++) {
            ker_sum += get_kernel_value(inker_data, si * 2 + sj, vwidth_kernel);
        }
    }

    if (strides >= 2) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
                LoadV_D_Main(col_ptr, 2, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 2, 1, 0);
                temp = Conv(2);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, 2, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height,   vwidth_main), 2, 1, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*2, vwidth_main);
            temp = Conv(2);
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, i, vwidth_main, handle_overflow(temp, vwidth_main));

            for (int j=1; j<width; j++) {
                LoadV_P(col_ptr, 2, 0);
                col_ptr = add_addr64(col_ptr, input_image->height, vwidth_main);
                temp = Conv(2);
                temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
                temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
                put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *convolution_k1(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 1) / strides + 1;
    img->height = (input_image->height - 1) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel = input_kernel->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);
    uint64_t *inker_data = (uint64_t *)(input_kernel->addr);

    int size = round_up_div(width * height * (vwidth_main >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    LoadV_D_Kernel(kernel_ptr,   1, 0, 0);

    int ker_sum = get_kernel_value(inker_data, 0, vwidth_kernel);

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, 1, 0, 0);
            temp = Conv(1);
            temp = temp - input_image->zero_point * ker_sum + input_kernel->bias;
            temp = temp * out_scale->scale / (input_image->scale * input_kernel->scale) + out_scale->zero_point;
            put_main_value(img_data, j * height + i, vwidth_main, handle_overflow(temp, vwidth_main));
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *Convolution_SC(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {

    assert((input_kernel->size <= input_image->width) && (input_kernel->size <= input_image->height));
    assert((input_kernel->size <= 5) && (input_kernel->size >= 1));
    assert(input_kernel->scale != 0);
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));
    assert((input_kernel->vwidth == 0x8) || (input_kernel->vwidth == 0x4) || (input_kernel->vwidth == 0x2) || (input_kernel->vwidth == 0x1));

    switch (input_kernel->size) {
        case 5:
            return convolution_k5(input_image, input_kernel, strides, out_scale);
        case 4:
            return convolution_k4(input_image, input_kernel, strides, out_scale);
        case 3:
            return convolution_k3(input_image, input_kernel, strides, out_scale);
        case 2:
            return convolution_k2(input_image, input_kernel, strides, out_scale);
        case 1:
            return convolution_k1(input_image, input_kernel, strides, out_scale);
        default:
            return NULL;
    }
}

//max pool
image_t *maxpooling_k5(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 5) / strides + 1;
    img->height = (input_image->height - 5) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 5) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 5, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 5, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 5, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 5, 3, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth), 5, 4, 0);
                temp = Pool_Max(5);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 5, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 5, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 5, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 5, 3, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth), 5, 4, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*5, vwidth);
            temp = Pool_Max(5);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 5, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Max(5);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *maxpooling_k4(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 4) / strides + 1;
    img->height = (input_image->height - 4) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 4) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 4, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 4, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 4, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 4, 3, 0);
                temp = Pool_Max(4);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 4, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 4, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 4, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 4, 3, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*4, vwidth);
            temp = Pool_Max(4);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 4, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Max(4);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *maxpooling_k3(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 3) / strides + 1;
    img->height = (input_image->height - 3) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 3) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 3, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 3, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 3, 2, 0);
                temp = Pool_Max(3);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 3, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 3, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 3, 2, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*3, vwidth);
            temp = Pool_Max(3);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 3, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Max(3);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *maxpooling_k2(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 2) / strides + 1;
    img->height = (input_image->height - 2) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 2) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 2, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 2, 1, 0);
                temp = Pool_Max(2);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 2, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 2, 1, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*2, vwidth);
            temp = Pool_Max(2);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                LoadV_P(col_ptr, 2, 0);
                col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                temp = Pool_Max(2);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *maxpooling_k1(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 1) / strides + 1;
    img->height = (input_image->height - 1) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
            LoadV_D_Main(col_ptr, 1, 0, 0);
            temp = Pool_Max(1);
            put_main_value(img_data, j * height + i, vwidth, temp);
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *MaxPooling_SC(image_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));

    switch (pool_size) {
        case 5:
            return maxpooling_k5(input_image, strides);
        case 4:
            return maxpooling_k4(input_image, strides);
        case 3:
            return maxpooling_k3(input_image, strides);
        case 2:
            return maxpooling_k2(input_image, strides);
        case 1:
            return maxpooling_k1(input_image, strides);
        default:
            return NULL;
    }
}

//avg pool
image_t *avgpooling_k5(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 5) / strides + 1;
    img->height = (input_image->height - 5) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 5) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 5, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 5, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 5, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 5, 3, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth), 5, 4, 0);
                temp = Pool_Avg(5);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 5, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 5, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 5, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 5, 3, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*4, vwidth), 5, 4, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*5, vwidth);
            temp = Pool_Avg(5);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 5, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Avg(5);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *avgpooling_k4(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 4) / strides + 1;
    img->height = (input_image->height - 4) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 4) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 4, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 4, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 4, 2, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 4, 3, 0);
                temp = Pool_Avg(4);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 4, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 4, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 4, 2, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*3, vwidth), 4, 3, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*4, vwidth);
            temp = Pool_Avg(4);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 4, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Avg(4);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *avgpooling_k3(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 3) / strides + 1;
    img->height = (input_image->height - 3) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 3) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 3, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 3, 1, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 3, 2, 0);
                temp = Pool_Avg(3);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 3, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 3, 1, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height*2, vwidth), 3, 2, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*3, vwidth);
            temp = Pool_Avg(3);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, 3, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                }
                temp = Pool_Avg(3);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *avgpooling_k2(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 2) / strides + 1;
    img->height = (input_image->height - 2) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    if (strides >= 2) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
                LoadV_D_Main(col_ptr, 2, 0, 0);
                LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 2, 1, 0);
                temp = Pool_Avg(2);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth);
            LoadV_D_Main(col_ptr, 2, 0, 0);
            LoadV_D_Main(add_addr64(col_ptr, input_image->height, vwidth), 2, 1, 0);
            col_ptr = add_addr64(col_ptr, input_image->height*2, vwidth);
            temp = Pool_Avg(2);
            put_main_value(img_data, i, vwidth, temp);

            for (int j=1; j<width; j++) {
                LoadV_P(col_ptr, 2, 0);
                col_ptr = add_addr64(col_ptr, input_image->height, vwidth);
                temp = Pool_Avg(2);
                put_main_value(img_data, j * height + i, vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *avgpooling_k1(image_t *input_image, int strides) {
    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = (input_image->width - 1) / strides + 1;
    img->height = (input_image->height - 1) / strides + 1;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = input_image->scale;
    img->zero_point = input_image->zero_point;

    int width = img->width;
    int height = img->height;
    uint8_t vwidth = img->vwidth;
    uint64_t *inimg_data = (uint64_t *)(input_image->addr);

    int size = round_up_div(width * height * (vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
    uint16_t temp;
    uint64_t col_ptr;

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth);
            LoadV_D_Main(col_ptr, 1, 0, 0);
            temp = Pool_Avg(1);
            put_main_value(img_data, j * height + i, vwidth, temp);
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *AvgPooling_SC(image_t *input_image, int pool_size, int strides) {

    assert((pool_size <= input_image->width) && (pool_size <= input_image->height));
    assert((pool_size <= 5) && (pool_size >= 1));
    assert(input_image->order == 1);
    assert((input_image->vwidth == 0x80) || (input_image->vwidth == 0x40) || (input_image->vwidth == 0x20) || (input_image->vwidth == 0x10));

    switch (pool_size) {
        case 5:
            return avgpooling_k5(input_image, strides);
        case 4:
            return avgpooling_k4(input_image, strides);
        case 3:
            return avgpooling_k3(input_image, strides);
        case 2:
            return avgpooling_k2(input_image, strides);
        case 1:
            return avgpooling_k1(input_image, strides);
        default:
            return NULL;
    }
}

//act
image_t *Activation_SC(image_t *input_image, char *algorithm, uint16_t zero_point) {

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

    uint64_t zero = 0;
    if (vwidth == 0x80) {
        for (int j=0; j<4; j++) {
            zero |= (uint64_t)zero_point << (j * 16);
        }
    }
    else if (vwidth == 0x40) {
        zero_point = zero_point & 0xff;
        for (int j=0; j<8; j++) {
            zero |= (uint64_t)zero_point << (j * 8);
        }
    }
    else if (vwidth == 0x20) {
        zero_point = zero_point & 0xf;
        for (int j=0; j<16; j++) {
            zero |= (uint64_t)zero_point << (j * 4);
        }
    }
    else if (vwidth == 0x10) {
        zero_point = zero_point & 0x3;
        for (int j=0; j<32; j++) {
            zero |= (uint64_t)zero_point << (j * 2);
        }
    }

    uint64_t vwidth_reg = vwidth;
    LoadV_Width((uint64_t)&vwidth_reg);

    for (int i=0; i<size; i++) {
        img_data[i] = Act(inimg_data[i], zero);
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *Convolution_SC_Inter(image_t *input_image, kernel_t *input_kernel, int strides, out_scale_t *out_scale) {

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
    uint64_t col_ptr;
    int temp;

    uint64_t vwidth_reg = vwidth_main | vwidth_kernel;
    LoadV_Width((uint64_t)&vwidth_reg);

    uint64_t kernel_ptr = get_addr64_kernel(inker_data, 0, vwidth_kernel);
    for (int i=0; i<input_kernel->size; i++) {
        LoadV_D_Kernel(kernel_ptr + input_kernel->size * i, input_kernel->size, i, 0);
    }

    int ker_sum = 0;
    for (int si=0; si<k; si++) {
        for (int sj=0; sj<k; sj++) {
            ker_sum += get_kernel_value(inker_data, si * k + sj, vwidth_kernel);
        }
    }

    if (strides >= input_kernel->size) {
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                col_ptr = get_addr64(inimg_data, j * strides * input_image->height + i * strides, vwidth_main);
                LoadV_D_Main(col_ptr, input_kernel->size, 0, 0);
                for (int l=1; l<input_kernel->size; l++) {
                    LoadV_D_Main(add_addr64(col_ptr, input_image->height * l, vwidth_main), input_kernel->size, l, 0);
                }
                temp = Conv(input_kernel->size);
                temp = temp - input_image->zero_point * ker_sum;
                temp = (temp > 32767) ? 32767 : ((temp < -32767) ? -32767 : temp);
                put_main_value(img_data, j * height + i, img->vwidth, temp);
            }
        }
    }
    else {
        for (int i=0; i<height; i++) {
            col_ptr = get_addr64(inimg_data, i * strides, vwidth_main);
            LoadV_D_Main(col_ptr, input_kernel->size, 0, 0);
            for (int l=1; l<input_kernel->size; l++) {
                LoadV_D_Main(add_addr64(col_ptr, input_image->height * l, vwidth_main), input_kernel->size, l, 0);
            }
            col_ptr = add_addr64(col_ptr, input_image->height*input_kernel->size, vwidth_main);
            temp = Conv(input_kernel->size);
            temp = temp - input_image->zero_point * ker_sum;
            temp = (temp > 32767) ? 32767 : ((temp < -32767) ? -32767 : temp);
            put_main_value(img_data, i, img->vwidth, temp);

            for (int j=1; j<width; j++) {
                for (int l=0; l<strides; l++) {
                    LoadV_P(col_ptr, input_kernel->size, 0);
                    col_ptr = add_addr64(col_ptr, input_image->height, vwidth_main);
                }
                temp = Conv(input_kernel->size);
                temp = temp - input_image->zero_point * ker_sum;
                temp = (temp > 32767) ? 32767 : ((temp < -32767) ? -32767 : temp);
                put_main_value(img_data, j * height + i, img->vwidth, temp);
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

//multi channel
image_mc_t *Convolution(image_mc_t *input_image, kernel_mc_t *input_kernel, int strides, out_scale_mc_t *out_scale) {

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
            img_tmp[j] = Convolution_SC_Inter(input_image->img[j], curr_kernel, strides, &(out_scale->scale[i]));
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

image_mc_t *MaxPooling(image_mc_t *input_image, int pool_size, int strides) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = MaxPooling_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mc_t *AvgPooling(image_mc_t *input_image, int pool_size, int strides) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = (input_image->width - pool_size) / strides + 1;
    img_mc->height = (input_image->height - pool_size) / strides + 1;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = AvgPooling_SC(input_image->img[i], pool_size, strides);
    }

    return img_mc;
}

image_mc_t *Activation(image_mc_t *input_image, char *algorithm, uint16_t zero_point) {

    image_mc_t *img_mc = (image_mc_t *)malloc(sizeof(image_mc_t));
    img_mc->width = input_image->width;
    img_mc->height = input_image->height;
    img_mc->channel = input_image->channel;
    img_mc->order = input_image->order;

    for (int i=0; i<input_image->channel; i++) {
        img_mc->img[i] = Activation_SC(input_image->img[i], algorithm, zero_point);
    }

    return img_mc;
}

//fully connected
image_t *Flatten(image_mc_t *input_image) {

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = 1;
    img->height = input_image->width * input_image->height * input_image->channel;
    img->vwidth = input_image->img[0]->vwidth;
    img->order = 1;
    img->scale = input_image->img[0]->scale;
    img->zero_point = input_image->img[0]->zero_point;

    int size = round_up_div(img->height * (img->vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);

    if (input_image->order == 0) {
        for (int c=0; c<input_image->channel; c++) {
            for (int i=0; i<input_image->height; i++) {
                for (int j=0; j<input_image->width; j++) {
                    uint16_t temp = get_main_value((uint64_t *)(input_image->img[c]->addr), i * input_image->width + j, input_image->img[c]->vwidth);
                    put_main_value(img_data, c * input_image->width * input_image->height + i * input_image->width + j, img->vwidth, temp);
                }
            }
        }
    }
    else {
        for (int c=0; c<input_image->channel; c++) {
            for (int j=0; j<input_image->width; j++) {
                for (int i=0; i<input_image->height; i++) {
                    uint16_t temp = get_main_value((uint64_t *)(input_image->img[c]->addr), j * input_image->height + i, input_image->img[c]->vwidth);
                    put_main_value(img_data, c * input_image->width * input_image->height + i * input_image->width + j, img->vwidth, temp);
                }
            }
        }
    }

    img->addr = (void *)img_data;
    return img;
}

image_t *Dense(image_t *input_image, fc_filter_t *fc_filter_array, int units, out_scale_t *out_scale) {

    assert(input_image->width == fc_filter_array[0].width);
    assert(input_image->height == fc_filter_array[0].height);
    assert(input_image->order == fc_filter_array[0].order);
    assert(fc_filter_array[0].scale != 0);

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->width = 1;
    img->height = units;
    img->vwidth = input_image->vwidth;
    img->order = input_image->order;
    img->scale = out_scale->scale;
    img->zero_point = out_scale->zero_point;

    int size = round_up_div(img->height * (img->vwidth >> 3), 64);
    uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);

    int width = input_image->width;
    int height = input_image->height;
    uint8_t vwidth_main = input_image->vwidth;
    uint8_t vwidth_kernel;
    uint64_t *in_addr_img = (uint64_t *)(input_image->addr);
    uint64_t *in_addr_ker;

    int temp = 0;
    int fc_sum = 0;
    for (int u=0; u<units; u++) {
        in_addr_ker = (uint64_t *)(fc_filter_array[u].addr);
        vwidth_kernel = fc_filter_array[u].vwidth;
        temp = 0;
        fc_sum = 0;
        
        for (int j=0; j<width; j++) {
            for (int i=0; i<height; i++) {
                int8_t fc_value = get_kernel_value(in_addr_ker, j * height + i, vwidth_kernel);
                temp += get_main_value(in_addr_img, j * height + i, vwidth_main) * fc_value;
                fc_sum += fc_value;
            }
        }

        temp = temp - input_image->zero_point * fc_sum + fc_filter_array[u].bias;
        temp = temp * out_scale->scale / (input_image->scale * fc_filter_array[u].scale) + out_scale->zero_point;
        temp = handle_overflow(temp, vwidth_main);
        put_main_value(img_data, u, vwidth_main, temp);
    }

    img->addr = (void *)img_data;
    return img;
}


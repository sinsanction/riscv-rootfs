#include "cnnapi.h"
#include "cnnapi_common.h"


// util
image_t *Transpose(image_t *input_image) {

  image_t *img = (image_t *)malloc(sizeof(image_t));
  img->width = input_image->width;
  img->height = input_image->height;
  img->vwidth = input_image->vwidth;
  img->scale = input_image->scale;
  img->zero_point = input_image->zero_point;

  int width = input_image->width;
  int height = input_image->height;
  uint8_t vwidth = input_image->vwidth;
  uint64_t *inimg_addr = (uint64_t *)(input_image->addr);

  int size = round_up_div(width * height * (vwidth >> 3), 64);
  uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  uint16_t temp;

  if (input_image->order == 0) {
    img->order = 1;
    for (int i=0; i<height; i++) {
      for (int j=0; j<width; j++) {
        temp = get_main_value(inimg_addr, i * width + j, vwidth);
        put_main_value(img_data, j * height + i, vwidth, temp);
      }
    }
  }
  else if (input_image->order == 1) {
    img->order = 0;
    for (int j=0; j<width; j++) {
      for (int i=0; i<height; i++) {
        temp = get_main_value(inimg_addr, j * height + i, vwidth);
        put_main_value(img_data, i * width + j, vwidth, temp);
      }
    }
  }

  img->addr = (void *)img_data;
  return img;
}

image_t *MergeImage(image_t *input_image_a, image_t *input_image_b) {

  assert(input_image_a->width == input_image_b->width);
  assert(input_image_a->height == input_image_b->height);
  assert(input_image_a->vwidth == input_image_b->vwidth);
  assert(input_image_a->order == input_image_b->order);

  int width = input_image_a->width;
  int height = input_image_b->height;
  uint8_t vwidth = input_image_a->vwidth;

  image_t *img = (image_t *)malloc(sizeof(image_t));
  img->width = width;
  img->height = height;
  img->vwidth = vwidth;
  img->order = input_image_a->order;
  img->scale = input_image_a->scale;
  img->zero_point = input_image_a->zero_point;

  int size = round_up_div(width * height * (vwidth >> 3), 64);
  uint64_t *img_data = (uint64_t *)malloc(sizeof(uint64_t) * size);
  uint64_t *in_addr_a = (uint64_t *)(input_image_a->addr);
  uint64_t *in_addr_b = (uint64_t *)(input_image_b->addr);
  uint16_t temp1, temp2;

  for (int j=0; j<width; j++) {
    for (int i=0; i<height; i++) {
      temp1 = get_main_value(in_addr_a, j * height + i, vwidth);
      temp2 = get_main_value(in_addr_b, j * height + i, vwidth);
      put_main_value(img_data, j * height + i, vwidth, temp1 + temp2);
    }
  }

  img->addr = (void *)img_data;
  return img;
}

void Rescale_SC(image_t *input_image, out_scale_t *out_scale) {

  assert(input_image->order == 1);

  int width = input_image->width;
  int height = input_image->height;
  uint8_t vwidth = input_image->vwidth;

  uint64_t *in_addr = (uint64_t *)(input_image->addr);
  uint16_t temp;

  for (int j=0; j<width; j++) {
    for (int i=0; i<height; i++) {
      temp = get_main_value(in_addr, j * height + i, vwidth);
      temp = re_scale(temp, input_image->scale, input_image->zero_point, out_scale->scale, out_scale->zero_point);
      put_main_value(in_addr, j * height + i, vwidth, handle_overflow(temp, vwidth));
    }
  }

  input_image->scale = out_scale->scale;
  input_image->zero_point = out_scale->zero_point;
}

void Rescale(image_mc_t *input_image, out_scale_mc_t *out_scale) {

  assert(input_image->order == 1);
  assert(input_image->channel == out_scale->channel);

  for (int i=0; i<input_image->channel; i++) {
    Rescale_SC(input_image->img[i], &(out_scale->scale[i]));
  }
}


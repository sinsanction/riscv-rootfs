#include "cnnapi_common.h"


// read low-bit data
inline uint8_t get_main_uint4(uint8_t *data, int i) {
  int j = i / 2;
  int r = i % 2;
  uint8_t data_i = (data[j] >> (r * 4)) & 0xf;
  return data_i;
}

inline uint8_t get_main_uint2(uint8_t *data, int i) {
  int j = i / 4;
  int r = i % 4;
  uint8_t data_i = (data[j] >> (r * 2)) & 0x3;
  return data_i;
}

inline int8_t get_kernel_int4(int8_t *kernel, int i) {
  int j = i / 2;
  int r = i % 2;
  int8_t kernel_i = (kernel[j] >> (r * 4)) & 0xf;
  if (kernel_i & 0x8) {
    kernel_i = kernel_i | 0xf0;
  }
  return kernel_i;
}

inline int8_t get_kernel_int2(int8_t *kernel, int i) {
  int j = i / 4;
  int r = i % 4;
  int8_t kernel_i = (kernel[j] >> (r * 2)) & 0x3;
  if (kernel_i & 0x2) {
    kernel_i = kernel_i | 0xfc;
  }
  return kernel_i;
}

inline int8_t get_kernel_int1(int8_t *kernel, int i) {
  int j = i / 8;
  int r = i % 8;
  int8_t kernel_i = (kernel[j] >> r) & 0x1;
  return kernel_i;
}

uint16_t get_main_value(uint64_t *img_addr, int i, uint8_t vwidth) {
  uint16_t res;
  if (vwidth == 0x10) {
    res = get_main_uint2((uint8_t *)img_addr, i);
  }
  else if (vwidth == 0x20) {
    res = get_main_uint4((uint8_t *)img_addr, i);
  }
  else if (vwidth == 0x40) {
    uint8_t *data = (uint8_t *)img_addr;
    res = data[i];
  }
  else { //vwidth == 0x80
    uint16_t *data = (uint16_t *)img_addr;
    res = data[i];
  }
  return res;
}

int8_t get_kernel_value(uint64_t *kernel_addr, int i, uint8_t vwidth) {
  int8_t res;
  if (vwidth == 0x1) {
    res = get_kernel_int1((int8_t *)kernel_addr, i);
  }
  else if (vwidth == 0x2) {
    res = get_kernel_int2((int8_t *)kernel_addr, i);
  }
  else if (vwidth == 0x4) {
    res = get_kernel_int4((int8_t *)kernel_addr, i);
  }
  else { //vwidth == 0x8
    int8_t *data = (int8_t *)kernel_addr;
    res = data[i];
  }
  return res;
}


// write low-bit data
inline void put_uint4(uint8_t *data, int i, uint8_t value) {
  int j = i / 2;
  int r = i % 2;

  if (r == 0) {
    data[j] = (data[j] & 0xf0) | value; 
  }
  else {
    data[j] = (data[j] & 0xf) | (value << 4); 
  }
}

inline void put_uint2(uint8_t *data, int i, uint8_t value) {
  int j = i / 4;
  int r = i % 4;

  if (r == 0) {
    data[j] = (data[j] & 0xfc) | value; 
  }
  else if (r == 1) {
    data[j] = (data[j] & 0xf3) | (value << 2); 
  }
  else if (r == 2) {
    data[j] = (data[j] & 0xcf) | (value << 4); 
  }
  else {
    data[j] = (data[j] & 0x3f) | (value << 6); 
  }
}

inline void put_uint1(uint8_t *data, int i, uint8_t value) {
  int j = i / 8;
  int r = i % 8;

  if (r == 0) {
    data[j] = (data[j] & 0xfe) | value; 
  }
  else if (r == 1) {
    data[j] = (data[j] & 0xfd) | (value << 1); 
  }
  else if (r == 2) {
    data[j] = (data[j] & 0xfb) | (value << 2); 
  }
  else if (r == 3) {
    data[j] = (data[j] & 0xf7) | (value << 3); 
  }
  else if (r == 4) {
    data[j] = (data[j] & 0xef) | (value << 4); 
  }
  else if (r == 5) {
    data[j] = (data[j] & 0xdf) | (value << 5); 
  }
  else if (r == 6) {
    data[j] = (data[j] & 0xbf) | (value << 6); 
  }
  else {
    data[j] = (data[j] & 0x7f) | (value << 7); 
  }
}

void put_main_value(uint64_t *img_addr, int i, uint8_t vwidth, uint16_t value) {
  if (vwidth == 0x10) {
    put_uint2((uint8_t *)img_addr, i, value);
  }
  else if (vwidth == 0x20) {
    put_uint4((uint8_t *)img_addr, i, value);
  }
  else if (vwidth == 0x40) {
    uint8_t *data = (uint8_t *)img_addr;
    data[i] = value;
  }
  else { //vwidth == 0x80
    uint16_t *data = (uint16_t *)img_addr;
    data[i] = value;
  }
}


// low-bit addr
uint64_t get_addr64(uint64_t *ptr, int i, uint8_t vwidth) {
    if (vwidth == 0x80) {
        return (uint64_t)ptr + (i << 1);
    }
    else if (vwidth == 0x40) {
        return (uint64_t)ptr + i;
    }
    else if (vwidth == 0x20) {
        return ((uint64_t)ptr << 1) + i;
    }
    else { //vwidth == 0x10
        return ((uint64_t)ptr << 2) + i;
    }
}

uint64_t add_addr64(uint64_t addr, int i, uint8_t vwidth) {
    if (vwidth == 0x80) {
        return addr + (i << 1);
    }
    else { //vwidth == 0x40 || vwidth == 0x20 || vwidth == 0x10
        return addr + i;
    }
}

uint64_t get_addr64_kernel(uint64_t *ptr, int i, uint8_t vwidth) {
    if (vwidth == 0x8) {
        return (uint64_t)ptr + i;
    }
    else if (vwidth == 0x4) {
        return ((uint64_t)ptr << 1) + i;
    }
    else if (vwidth == 0x2) {
        return ((uint64_t)ptr << 2) + i;
    }
    else { //(vwidth == 0x1)
        return ((uint64_t)ptr << 3) + i;
    }
}


// others
inline int round_up_div(int a, int b) {
  int div = a / b;
  int rem = a % b;
  if (rem == 0) {
    return div;
  }
  else {
    return div + 1;
  }
}

uint16_t handle_overflow(int32_t tmp, uint8_t vwidth) {
  tmp = (tmp < 0) ? 0 : tmp;
  if (vwidth == 0x80) {
      return (tmp > 65535) ? 65535 : tmp;
  }
  else if (vwidth == 0x40) {
      return (tmp > 255) ? 255 : tmp;
  }
  else if (vwidth == 0x20) {
      return (tmp > 15) ? 15 : tmp;
  }
  else { //vwidth == 0x10
      return (tmp > 3) ? 3 : tmp;
  }
}

inline int re_scale(int old_value, uint16_t old_scale, uint16_t old_zero, uint16_t new_scale, uint16_t new_zero) {
  int new_value = (old_value - old_zero) * new_scale / old_scale + new_zero;
  return new_value;
}

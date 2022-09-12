#ifndef __CNNAPI_COMMON_H__
#define __CNNAPI_COMMON_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// read low-bit data
uint16_t get_main_value(uint64_t *img_addr, int i, uint8_t vwidth);

int8_t get_kernel_value(uint64_t *kernel_addr, int i, uint8_t vwidth);

// write low-bit data
void put_main_value(uint64_t *img_addr, int i, uint8_t vwidth, uint16_t value);

// low-bit addr
uint64_t get_addr64(uint64_t *ptr, int i, uint8_t vwidth);

uint64_t add_addr64(uint64_t addr, int i, uint8_t vwidth);

uint64_t get_addr64_kernel(uint64_t *ptr, int i, uint8_t vwidth);

// others
int round_up_div(int a, int b);

uint16_t handle_overflow(int32_t tmp, uint8_t vwidth);

int re_scale(int old_value, uint16_t old_scale, uint16_t old_zero, uint16_t new_scale, uint16_t new_zero);

#ifdef __cplusplus
}
#endif

#endif

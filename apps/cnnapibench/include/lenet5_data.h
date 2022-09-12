#ifndef __LENET5_DATA_H__
#define __LENET5_DATA_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uint8_t;
typedef signed char int8_t;

#define IMAGE_0_NUM 980
#define IMAGE_1_NUM 1135
#define IMAGE_2_NUM 1032
#define IMAGE_3_NUM 1010
#define IMAGE_4_NUM 982
#define IMAGE_5_NUM 892
#define IMAGE_6_NUM 958
#define IMAGE_7_NUM 1028
#define IMAGE_8_NUM 974
#define IMAGE_9_NUM 1009

void remap_data();
void remap_image0();
void remap_image1();
void remap_image2();
void remap_image3();
void remap_image4();
void remap_image5();
void remap_image6();
void remap_image7();
void remap_image8();
void remap_image9();

// input image
extern uint8_t *Input_Image[10];

// c1 kernel
extern int8_t *C1_Kernel;  //[6][5 * 5]
extern int *C1_Bias;  //[6]

// c3 kernel
extern int8_t *C3_Kernel;  //[16 * 6][5 * 5]
extern int *C3_Bias;  //[16]

// fc1 filter
extern int8_t *FC1_Filter;  //[120][256]
extern int *FC1_Bias;  //[120]

// fc2 filter
extern int8_t *FC2_Filter;  //[84][120]
extern int *FC2_Bias;  //[84]

// fc3 filter
extern int8_t *FC3_Filter;  //[10][84]
extern int *FC3_Bias;  //[10]

#ifdef __cplusplus
}
#endif

#endif

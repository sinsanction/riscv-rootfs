#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include<time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MB * 1024 * 1024
#define KB * 1024

#define REPEAT  1

//                  size |  heap  |  checksum
#define CONV1   {     100, 128 KB,  0x00000001}
#define CONV2   {     100, 128 KB,  0x00000002}
#define CONV3   {     100, 128 KB,  0x00000003}
#define CONV4   {     100, 128 KB,  0x00000004}

#define POOL1   {     100, 128 KB,  0x00000011}
#define POOL2   {     100, 128 KB,  0x00000012}
#define POOL3   {     100, 128 KB,  0x00000013}
#define POOL4   {     100, 128 KB,  0x00000014}

#define ACT1    {     100, 128 KB,  0x00000021}
#define ACT2    {     100, 128 KB,  0x00000022}

//api v2.0
#define CONV1_V2   {     100, 128 KB,  0x00020001}
#define CONV2_V2   {     100, 128 KB,  0x00020002}
#define CONV3_V2   {     100, 128 KB,  0x00020003}
#define CONV4_V2   {     100, 128 KB,  0x00020004}

#define POOL1_V2   {     100, 128 KB,  0x00020011}
#define POOL2_V2   {     100, 128 KB,  0x00020012}
#define POOL3_V2   {     100, 128 KB,  0x00020013}
#define POOL4_V2   {     100, 128 KB,  0x00020014}

#define ACT1_V2    {     100, 128 KB,  0x00020021}
#define ACT2_V2    {     100, 128 KB,  0x00020022}

//lenet
#define LENET5        {     100, 256 KB,  0x00030001}
#define LENET5_BASE   {     100, 128 KB,  0x00030002}
#define LENET5_PERF   {     100, 128 KB,  0x00030003}

#define BENCHMARK_LIST(def) \
  def(conv1, "conv1", CONV1, "CNN Conv 1") \
  def(conv2, "conv2", CONV2, "CNN Conv 2") \
  def(conv3, "conv3", CONV3, "CNN Conv 3 multi channel") \
  def(conv4, "conv4", CONV4, "CNN Conv 4 multi channel") \
  def(pool1, "pool1", POOL1, "CNN Pool 1") \
  def(pool2, "pool2", POOL2, "CNN Pool 2") \
  def(pool3, "pool3", POOL3, "CNN Pool 3 multi channel") \
  def(pool4, "pool4", POOL4, "CNN Pool 4 multi channel") \
  def(act1,  "act1",  ACT1,  "CNN Act 1") \
  def(act2,  "act2",  ACT2,  "CNN Act 2 multi channel")

#define BENCHMARK_V2_LIST(def) \
  def(conv1_v2, "conv1_v2", CONV1_V2, "CNNv2.0 Conv 1") \
  def(conv2_v2, "conv2_v2", CONV2_V2, "CNNv2.0 Conv 2") \
  def(conv3_v2, "conv3_v2", CONV3_V2, "CNNv2.0 Conv 3 multi channel") \
  def(conv4_v2, "conv4_v2", CONV4_V2, "CNNv2.0 Conv 4 multi channel") \
  def(pool1_v2, "pool1_v2", POOL1_V2, "CNNv2.0 Pool 1") \
  def(pool2_v2, "pool2_v2", POOL2_V2, "CNNv2.0 Pool 2") \
  def(pool3_v2, "pool3_v2", POOL3_V2, "CNNv2.0 Pool 3 multi channel") \
  def(pool4_v2, "pool4_v2", POOL4_V2, "CNNv2.0 Pool 4 multi channel") \
  def(act1_v2,  "act1_v2",  ACT1_V2,  "CNNv2.0 Act 1") \
  def(act2_v2,  "act2_v2",  ACT2_V2,  "CNNv2.0 Act 2 multi channel")

#define BENCHMARK_LENET_LIST(def) \
  def(lenet5,      "lenet5",      LENET5,      "CNN LeNet5") \
  def(lenet5_base, "lenet5_base", LENET5_BASE, "CNN LeNet5 base") \
  def(lenet5_perf, "lenet5_perf", LENET5_PERF, "CNN LeNet5 perf")

// Each benchmark will run REPEAT times

#define DECL(_name, _sname, _setting, _desc) \
  void bench_##_name##_prepare(); \
  void bench_##_name##_run(); \
  int bench_##_name##_validate();

BENCHMARK_LIST(DECL)
BENCHMARK_V2_LIST(DECL)
BENCHMARK_LENET_LIST(DECL)

typedef struct Setting {
  int size;
  unsigned long mlim;
  uint32_t checksum;
} Setting;

typedef struct Benchmark {
  void (*prepare)();
  void (*run)();
  int (*validate)();
  const char *name, *desc;
  Setting settings;
} Benchmark;

extern Benchmark *current;
extern Setting *setting;

typedef struct Result {
  int pass;
  unsigned long clk;
} Result;

void prepare(Result *res);
void done(Result *res);

// random number generator
void bench_srand(uint32_t seed);
uint32_t bench_rand(); // return a random number between 0..32767

// checksum
uint32_t checksum(void *start, void *end);

#ifdef __cplusplus
}
#endif

#endif

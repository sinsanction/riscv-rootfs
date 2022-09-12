#include <benchmark.h>

Benchmark *current;
Setting *setting;

// The benchmark list

#define ENTRY(_name, _sname, _setting, _desc) \
  { .prepare = bench_##_name##_prepare, \
    .run = bench_##_name##_run, \
    .validate = bench_##_name##_validate, \
    .name = _sname, \
    .desc = _desc, \
    .settings = _setting, },

Benchmark benchmarks_func[] = {
  BENCHMARK_LIST(ENTRY)
};
Benchmark benchmarks_func_v2[] = {
  BENCHMARK_V2_LIST(ENTRY)
};
Benchmark benchmarks_lenet[] = {
  BENCHMARK_LENET_LIST(ENTRY)
};

// Running a benchmark
static void bench_prepare(Result *res) {
  res->clk = clock();
}

static void bench_done(Result *res) {
  res->clk = clock() - res->clk;
}

static void run_once(Benchmark *b, Result *res) {
  //bench_reset();       // reset malloc state
  current->prepare();  // call bechmark's prepare function
  bench_prepare(res);  // clean everything, start timer
  current->run();      // run it
  bench_done(res);     // collect results
  res->pass = current->validate();
}

int main() {
  const char setting_name[] = "test_lenet_func";
  int setting_id = -1;

  if      (strcmp(setting_name, "testfunc") == 0) setting_id = 0;
  else if (strcmp(setting_name, "testfuncv2") == 0) setting_id = 1;
  else if (strcmp(setting_name, "test_lenet_func") == 0) setting_id = 2;
  else if (strcmp(setting_name, "test_lenet_base") == 0) setting_id = 3;
  else if (strcmp(setting_name, "test_lenet_perf") == 0) setting_id = 4;
  else if (strcmp(setting_name, "test_lenet_real_func") == 0) setting_id = 5;
  else if (strcmp(setting_name, "test_lenet_real_perf") == 0) setting_id = 7;
  else {
    printf("Invalid mainargs: \"%s\"; "
           "must be in {testfunc, testfuncv2}\n", setting_name);
    return 1;
  }

  printf("======= Running CNNAPIMicroBench [input *%s* id:%d] =======\n", setting_name, setting_id);

  int pass = 1;
  unsigned long t0 = clock();
  int bench_length;
  Benchmark *test_bench;
  if      (setting_id == 1) { bench_length = LENGTH(benchmarks_func_v2); test_bench = benchmarks_func_v2; }
  else if (setting_id == 2) { bench_length = 1; test_bench = &benchmarks_lenet[0]; }
  else if (setting_id == 3) { bench_length = 1; test_bench = &benchmarks_lenet[1]; }
  else if (setting_id == 4) { bench_length = 1; test_bench = &benchmarks_lenet[2]; }
  else if (setting_id == 5) { bench_length = 1; test_bench = &benchmarks_lenet[3]; }
  else if (setting_id == 7) { bench_length = 1; test_bench = &benchmarks_lenet[4]; }
  else { bench_length = LENGTH(benchmarks_func); test_bench = benchmarks_func; }

  for (int i = 0; i < bench_length; i ++) {
    Benchmark *bench = &test_bench[i];
    current = bench;
    setting = &bench->settings;
    printf("\n[%s] %s: \n", bench->name, bench->desc);
  
    unsigned long clk = ULONG_MAX;
    int succ = 1;
    for (int i = 0; i < REPEAT; i ++) {
      Result res;
      run_once(bench, &res);
      printf(res.pass ? "*" : "X");
      succ &= res.pass;
      if (res.clk < clk) clk = res.clk;
    }

    if (succ) printf(" Passed.");
    else printf(" Failed.");

    pass &= succ;

    printf("\n");
    unsigned int msec = clk * 1000 / CLOCKS_PER_SEC;
    printf("  min time: %d clock, %d ms\n", clk, msec);
  }
  unsigned long t1 = clock();

  printf("==================================================\n");
  printf("CNNAPIBench %s", pass ? "PASS" : "FAIL");
  printf("\n");
  printf("Total time: %d clock, %d ms\n", t1 - t0, (t1 - t0) * 1000 / CLOCKS_PER_SEC);
  return !pass;
}

// Libraries
static uint32_t seed = 1;

void bench_srand(uint32_t _seed) {
  seed = _seed & 0x7fff;
}

uint32_t bench_rand() {
  seed = (seed * (uint32_t)214013L + (uint32_t)2531011L);
  return (seed >> 16) & 0x7fff;
}

// FNV hash
uint32_t checksum(void *start, void *end) {
  const uint32_t x = 16777619;
  uint32_t h1 = 2166136261u;
  for (uint8_t *p = (uint8_t*)start; p + 4 < (uint8_t*)end; p += 4) {
    for (int i = 0; i < 4; i ++) {
      h1 = (h1 ^ p[i]) * x;
    }
  }
  int32_t hash = (uint32_t)h1;
  hash += hash << 13;
  hash ^= hash >> 7;
  hash += hash << 3;
  hash ^= hash >> 17;
  hash += hash << 5;
  return hash;
}

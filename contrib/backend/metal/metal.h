#ifndef GOML_METAL_H
#define GOML_METAL_H
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint32_t n, rank, op, flags;
    uint32_t shape[8], as[8], bs[8];
    uint32_t ao, bo;
    uint32_t ints[8];
    float scalars[4];
} GMParams;

void *gm_open(int index, const char *source, char **error);

const char *gm_name(void *device);

void *gm_allocate(void *device, uint64_t bytes, char **error);

void gm_release_buffer(void *buffer);

int gm_upload(void *device, void *buffer, const float *data, uint64_t bytes, char **error);

int gm_download(void *device, void *buffer, float *data, uint64_t bytes, char **error);

int gm_compute(void *device, const char *name, void *a, void *b, void *out, GMParams *params,
               char **error);

int gm_matmul(void *device, void *a, void *b, void *out, uint32_t m, uint32_t k, uint32_t n,
              uint32_t ao, uint32_t bo, uint32_t ar, uint32_t br, int ta, int tb, char **error);

int gm_conv(void *device, void *input, void *weight, void *gradient, void *out, void *dw,
            GMParams *params, int backward, char **error);

int gm_sync(void *device, char **error);

void gm_close(void *device);

void gm_stats(void *device, uint64_t *live, uint64_t *peak);

#endif

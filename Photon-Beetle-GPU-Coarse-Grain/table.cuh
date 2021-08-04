#include <stdint.h>

__global__ void crypto_hash_gpu_table(uint8_t *out,  uint8_t *in,  uint32_t inlen);
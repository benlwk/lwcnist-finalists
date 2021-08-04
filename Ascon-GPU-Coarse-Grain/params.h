#ifndef PARAMS_H
#define PARAMS_H


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// GPU
#define MAX_MESSAGE_LENGTH 1024
#define BATCH 1*1024*1024
#define MLEN 1024
#define BLOCK_SIZE 512
#define PERF
// #define DEBUG


// ASCON
#define STATE_SIZE 64*5
#define CRYPTO_BYTES 32
#define ASCON_HASH_OUTLEN 32 /* HASH */
#define ASCON_HASH_ROUNDS 12


typedef struct {
  uint64_t x0, x1, x2, x3, x4;
} state_t;
#endif
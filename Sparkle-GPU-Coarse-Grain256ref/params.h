#ifndef PARAMS_H
#define PARAMS_H


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// GPU
#define MAX_MESSAGE_LENGTH 1024
#define BATCH 4*1024*1024
#define MLEN 1024
#define BLOCK_SIZE 512
#define PERF
// #define DEBUG
#define OPT

// sparkle

// Define the ESCH instance here (api.h has to match!). The main instance is
// ESCH256, which has a block size of 128 bits and a digest size of 256 bits.
// Another instance of ESCH is ESCH384.
#define CRYPTO_BYTES 32
#define ESCH256

///////////////////
#if defined ESCH256
///////////////////

#define ESCH_DIGEST_LEN     256

#define SPARKLE_STATE       384
#define SPARKLE_RATE        128
#define SPARKLE_CAPACITY    256

#define SPARKLE_STEPS_SLIM  7
#define SPARKLE_STEPS_BIG   11


/////////////////////
#elif defined ESCH384
/////////////////////

#define ESCH_DIGEST_LEN     384

#define SPARKLE_STATE       512
#define SPARKLE_RATE        128
#define SPARKLE_CAPACITY    384

#define SPARKLE_STEPS_SLIM  8
#define SPARKLE_STEPS_BIG   12


#else
#error "Invalid definition of ESCH instance."
#endif
typedef struct {
  uint64_t x0, x1, x2, x3, x4;
} state_t;
#endif
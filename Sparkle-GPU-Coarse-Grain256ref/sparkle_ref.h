///////////////////////////////////////////////////////////////////////////////
// sparkle_ref.h: Reference C99 implementation of the SPARKLE permutation.   //
// This file is part of the SPARKLE submission to NIST's LW Crypto Project.  //
// Version 1.1.2 (2020-10-30), see <http://www.cryptolux.org/> for updates.  //
// Authors: The SPARKLE Group (C. Beierle, A. Biryukov, L. Cardoso dos       //
// Santos, J. Groszschaedl, L. Perrin, A. Udovenko, V. Velichkov, Q. Wang).  //
// License: GPLv3 (see LICENSE file), other licenses available upon request. //
// Copyright (C) 2019-2020 University of Luxembourg <http://www.uni.lu/>.    //
// ------------------------------------------------------------------------- //
// This program is free software: you can redistribute it and/or modify it   //
// under the terms of the GNU General Public License as published by the     //
// Free Software Foundation, either version 3 of the License, or (at your    //
// option) any later version. This program is distributed in the hope that   //
// it will be useful, but WITHOUT ANY WARRANTY; without even the implied     //
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  //
// GNU General Public License for more details. You should have received a   //
// copy of the GNU General Public License along with this program. If not,   //
// see <http://www.gnu.org/licenses/>.                                       //
///////////////////////////////////////////////////////////////////////////////

#ifndef SPARKLE_REF_H
#define SPARKLE_REF_H

#include <stdint.h>
#include "params.h"

#define MAX_BRANCHES 8

typedef struct {
  uint32_t x[MAX_BRANCHES];
  uint32_t y[MAX_BRANCHES];
} SparkleState;

typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

#define DIGEST_WORDS (ESCH_DIGEST_LEN/32)
#define DIGEST_BYTES (ESCH_DIGEST_LEN/8)
#define STATE_BRANS  (SPARKLE_STATE/64)
#define STATE_WORDS  (SPARKLE_STATE/32)
#define STATE_BYTES  (SPARKLE_STATE/8)
#define RATE_BRANS   (SPARKLE_RATE/64)
#define RATE_WORDS   (SPARKLE_RATE/32)
#define RATE_BYTES   (SPARKLE_RATE/8)
#define CAP_BRANS    (SPARKLE_CAPACITY/64)
#define CAP_WORDS    (SPARKLE_CAPACITY/32)
#define CAP_BYTES    (SPARKLE_CAPACITY/8)
#define CONST_M1 (((uint32_t) 1) << 24)
#define CONST_M2 (((uint32_t) 2) << 24)
#define ROT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define ELL(x) (ROT(((x) ^ ((x) << 16)), 16))



// 4-round ARX-box
#define ARXBOX(x, y, c)                     \
  (x) += ROT((y), 31), (y) ^= ROT((x), 24), \
  (x) ^= (c),                               \
  (x) += ROT((y), 17), (y) ^= ROT((x), 17), \
  (x) ^= (c),                               \
  (x) += (y),          (y) ^= ROT((x), 31), \
  (x) ^= (c),                               \
  (x) += ROT((y), 24), (y) ^= ROT((x), 16), \
  (x) ^= (c)


// Inverse of 4-round ARX-box
#define ARXBOX_INV(x, y, c)                 \
  (x) ^= (c),                               \
  (y) ^= ROT((x), 16), (x) -= ROT((y), 24), \
  (x) ^= (c),                               \
  (y) ^= ROT((x), 31), (x) -= (y),          \
  (x) ^= (c),                               \
  (y) ^= ROT((x), 17), (x) -= ROT((y), 17), \
  (x) ^= (c),                               \
  (y) ^= ROT((x), 24), (x) -= ROT((y), 31)


// Round constants
__constant__ static uint32_t RCON[MAX_BRANCHES] = {      \
  0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, \
  0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D  \
};


__device__ void linear_layer(SparkleState *state, int brans)
{
  int i, b = brans/2;
  uint32_t *x = state->x, *y = state->y;
  uint32_t tmp;
  
  // Feistel function (adding to y part)
  tmp = 0;
  for(i = 0; i < b; i++)
    tmp ^= x[i];
  tmp = ELL(tmp);
  for(i = 0; i < b; i ++)
    y[i+b] ^= (tmp ^ y[i]);
  
  // Feistel function (adding to x part)
  tmp = 0;
  for(i = 0; i < b; i++)
    tmp ^= y[i];
  tmp = ELL(tmp);
  for(i = 0; i < b; i ++)
    x[i+b] ^= (tmp ^ x[i]);
  
  // Branch swap with 1-branch left-rotation of right side
  // <------- left side --------> <------- right side ------->
  //    0    1    2 ...  B-2  B-1    B  B+1  B+2 ... 2B-2 2B-1
  //  B+1  B+2  B+3 ... 2B-1    B    0    1    2 ...  B-2  B-1
  
  // Branch swap of the x part
  tmp = x[0];
  for (i = 0; i < b - 1; i++) {
    x[i] = x[i+b+1];
    x[i+b+1] = x[i+1];
  }
  x[b-1] = x[b];
  x[b] = tmp;
  
  // Branch swap of the y part
  tmp = y[0];
  for (i = 0; i < b - 1; i++) {
    y[i] = y[i+b+1];
    y[i+b+1] = y[i+1];
  }
  y[b-1] = y[b];
  y[b] = tmp;
}


__device__ void sparkle_ref(SparkleState *state, int brans, int steps)
{
  int i, j;  // Step and branch counter
  
  // The number of branches must be even and not bigger than MAX_BRANCHES.
  // assert(((brans & 1) == 0) && (brans >= 4) && (brans <= MAX_BRANCHES));
  
  for(i = 0; i < steps; i++) {
    // Add step counter
    state->y[0] ^= RCON[i%MAX_BRANCHES];
    state->y[1] ^= i;
    // ARXBox layer
    for(j = 0; j < brans; j ++)
      ARXBOX(state->x[j], state->y[j], RCON[j]);
    // Linear layer
    linear_layer(state, brans);
  }
}


__device__ void linear_layer_inv(SparkleState *state, int brans)
{
  int i, b = brans/2;
  uint32_t *x = state->x, *y = state->y;
  uint32_t tmp;
  
  // Branch swap with 1-branch right-rotation of left side
  // <------- left side --------> <------- right side ------->
  //    0    1    2 ...  B-2  B-1    B  B+1  B+2 ... 2B-2 2B-1
  //    B  B+1  B+2 ... 2B-2 2B-1  B-1    0    1 ...  B-3  B-2
  
  // Branch swap of the x part
  tmp = x[b-1];
  for (i = b - 1; i > 0; i--) {
    x[i] = x[i+b];
    x[i+b] = x[i-1];
  }
  x[0] = x[b];
  x[b] = tmp;
  
  // Branch swap of the y part
  tmp = y[b-1];
  for (i = b - 1; i > 0; i--) {
    y[i] = y[i+b];
    y[i+b] = y[i-1];
  }
  y[0] = y[b];
  y[b] = tmp;
  
  // Feistel function (adding to x part)
  tmp = 0;
  for(i = 0; i < b; i ++)
    tmp ^= y[i];
  tmp = ELL(tmp);
  for(i = 0; i < b; i ++)
    x[i+b] ^= (tmp ^ x[i]);
  
  // Feistel function (adding to y part)
  tmp = 0;
  for(i = 0; i < b; i ++)
    tmp ^= x[i];
  tmp = ELL(tmp);
  for(i = 0; i < b; i ++)
    y[i+b] ^= (tmp ^ y[i]);
}


__device__ void sparkle_inv_ref(SparkleState *state, int brans, int steps)
{
  int i, j;  // Step and branch counter
  
  // The number of branches must be even and not bigger than MAX_BRANCHES.
  // assert(((brans & 1) == 0) && (brans >= 4) && (brans <= MAX_BRANCHES));
  
  for(i = steps - 1; i >= 0; i--) {
    // Linear layer
    linear_layer_inv(state, brans);
    // ARXbox layer
    for(j = 0; j < brans; j ++)
      ARXBOX_INV(state->x[j], state->y[j], RCON[j]);
    // Add step counter
    state->y[1] ^= i;
    state->y[0] ^= RCON[i%MAX_BRANCHES];
  }
}

///////////////////////////////////////////////////////////////////////////////
/////// HELPER FUNCTIONS AND MACROS (INJECTION OF MESSAGE BLOCK, ETC.) ////////
///////////////////////////////////////////////////////////////////////////////


// Injection of a 16-byte block of the message to the state.

__device__ static void add_msg_blk(SparkleState *state, const uint8_t *in, size_t inlen)
{
  uint32_t buffer[STATE_WORDS/2] = { 0 };
  uint32_t tmpx = 0, tmpy = 0;
  int i;
  
  memcpy(buffer, in, inlen);
  if (inlen < RATE_BYTES)  // padding
    *(((uint8_t *) buffer) + inlen) = 0x80;
  
  // Feistel function part 1: computation of ELL(tmpx) and ELL(tmpy)
  for(i = 0; i < (STATE_WORDS/2); i += 2) {
    tmpx ^= buffer[i];
    tmpy ^= buffer[i+1];
  }
  tmpx = ELL(tmpx);
  tmpy = ELL(tmpy);
  // Feistel function part 2: state is XORed with tmpx/tmpy and msg
  for(i = 0; i < (STATE_BRANS/2); i++) {
    state->x[i] ^= (buffer[2*i] ^ tmpy);
    state->y[i] ^= (buffer[2*i+1] ^ tmpx);
  }
}


///////////////////////////////////////////////////////////////////////////////
///////////// LOW-LEVEL HASH FUNCTIONS (FOR USE WITH FELICS-HASH) /////////////
///////////////////////////////////////////////////////////////////////////////


// The Initialize function sets all branches of the state to 0.

__device__ void Initialize(SparkleState *state)
{
  int i;
  
  for (i = 0; i < STATE_BRANS; i++)
    state->x[i] = state->y[i] = 0;
}


// The ProcessMessage function absorbs the message into the state (in blocks of 16 bytes). According to the specification, the constant Const_M is first transformed via the inverse Feistel function, added to the (padded) message block, and finally injected to the state via the Feistel function. Since the Feistel function and the inverse Feistel function cancel out, we can simply inject the constant directly to the state.

__device__ void ProcessMessage(SparkleState *state, const UChar *in, size_t inlen)
{
  // Main Hashing Loop
  
  while (inlen > RATE_BYTES) {
    // addition of a message block to the state
    add_msg_blk(state, in, RATE_BYTES);
    // execute SPARKLE with slim number of steps
    sparkle_ref(state, STATE_BRANS, SPARKLE_STEPS_SLIM);
    inlen -= RATE_BYTES;
    in += RATE_BYTES;
  }
  
  // Hashing of Last Block
  
  // addition of constant M1 or M2 to the state
  state->y[(STATE_BRANS/2)-1] ^= ((inlen < RATE_BYTES) ? CONST_M1 : CONST_M2);
  // addition of last msg block (incl. padding)
  add_msg_blk(state, in, inlen);
  // execute SPARKLE with big number of steps
  sparkle_ref(state, STATE_BRANS, SPARKLE_STEPS_BIG);
}


// The Finalize function generates the message digest by "squeezing" (i.e. by calling SPARKLE with a slim number of steps) until the digest has reached a byte-length of DIGEST_BYTES.

__device__ void Finalize(SparkleState *state, UChar *out)
{
  uint32_t buffer[DIGEST_WORDS];
  int i, outlen = 0;
  
  for (i = 0; i < RATE_BRANS; i++) {
    buffer[outlen++] = state->x[i];
    buffer[outlen++] = state->y[i];
  }
  while (outlen < DIGEST_WORDS) {
    sparkle_ref(state, STATE_BRANS, SPARKLE_STEPS_SLIM);
    for (i = 0; i < RATE_BRANS; i++) {
      buffer[outlen++] = state->x[i];
      buffer[outlen++] = state->y[i];
    }
  }
  memcpy(out, buffer, DIGEST_BYTES);
}

#endif  // SPARKLE_REF_H


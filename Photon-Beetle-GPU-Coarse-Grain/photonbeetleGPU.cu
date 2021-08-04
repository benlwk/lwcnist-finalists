#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <time.h>  
#include <sys/time.h>
#include "params.h"
#include "table.cuh"
#include "bitslice_sb32.cuh"
#include "bitslice_sb64.cuh"

// const uint8_t ReductionPoly = 0x3;
const uint8_t WORDFILTER = ((uint8_t) 1<<S)-1;

__constant__ uint8_t RC[D][12] = {
  {1, 3, 7, 14, 13, 11, 6, 12, 9, 2, 5, 10},
  {0, 2, 6, 15, 12, 10, 7, 13, 8, 3, 4, 11},
  {2, 0, 4, 13, 14, 8, 5, 15, 10, 1, 6, 9},
  {6, 4, 0, 9, 10, 12, 1, 11, 14, 5, 2, 13},
  {14, 12, 8, 1, 2, 4, 9, 3, 6, 13, 10, 5},
  {15, 13, 9, 0, 3, 5, 8, 2, 7, 12, 11, 4},
  {13, 15, 11, 2, 1, 7, 10, 0, 5, 14, 9, 6},
  {9, 11, 15, 6, 5, 3, 14, 4, 1, 10, 13, 2}
};

__constant__ uint32_t Table[128] = {0xbf5c86f8, 0xa9756b96, 0xceb643e4, 0x5dab3cd3, 0x1629ed6e, 0x0, 0x71eac51c, 0x931d7f37, 0x67c32872, 0xf4de5745, 0xd89fae8a, 0x3a6814a1, 0x85349259, 0xe2f7ba2b, 0x2c41f9cf, 0x4b82d1bd, 0x565ef4bc, 0x7b7d93a5, 0xb3b7e2c6, 0xacafd85b, 0x2d236719, 0x0, 0xe5e9167a, 0x1f183a9d, 0xc8ca7163, 0xd7d24bfe, 0x9e9485df, 0x6465a938, 0x323b5d84, 0xfaf12ce7, 0x4946ce21, 0x818cbf42, 0xba3969b3, 0xaec2b2ac, 0xc58d3dc8, 0x5761c156, 0x14fbdb1f, 0x0, 0x7fb4547b, 0x92ecfc9e, 0x6b4f8f64, 0xf9a373fa, 0xd176e6d7, 0x3c2e4e32, 0x86172781, 0xed58a8e5, 0x28d5952d, 0x439a1a49, 0xd33c3811, 0x1cc5c644, 0xf8868499, 0x966b6322, 0xcff9fe55, 0x0, 0x2bbabc88, 0x6eede7bb, 0xe44342dd, 0x8aaea566, 0x377f7acc, 0x722821ff, 0xa11419ee, 0x45575b33, 0xbdd1dfaa, 0x59929d77, 0xb26f4579, 0xa8b937f2, 0xc13e2bad, 0x54cd8ae1, 0x1ad6728b, 0x0, 0x73516ed4, 0x95f3a14c, 0x69871c5f, 0xfc74bd13, 0xdbe85926, 0x3d4a96be, 0x8f25d3c7, 0xe6a2cf98, 0x279ce435, 0x4e1bf86a, 0xa2539fc1, 0xe87c2954, 0x51b8de69, 0x74a61db2, 0x4a2fb695, 0x0, 0xf3eb41a8, 0x251ec3db, 0xb9c4f73d, 0x9cda34e6, 0x1b9768fc, 0xcd62ea8f, 0x6f31754e, 0xd6f58273, 0x874d5c1a, 0x3e89ab27, 0x993846cb, 0x22c63b5a, 0xdd84236c, 0x11638cb5, 0xbbfe7d91, 0x0, 0x44bc65a7, 0xcce7afd9, 0xff421836, 0x33a5b7ef, 0x667a5efd, 0xee219483, 0x7719d248, 0x885bca7e, 0x55dfe912, 0xaa9df124, 0xeb643e47, 0xdab3cd3f, 0x7c32872a, 0xf5c86f8e, 0x31d7f378, 0x0, 0x9756b96d, 0x89fae8a4, 0xa6814a15, 0x2f7ba2b1, 0x4de57452, 0x5349259b, 0xb82d1bdc, 0x1eac51c9, 0x629ed6e3, 0xc41f9cf6};

__device__ void PrintState(uint8_t state[D][D])
{  
  int i, j;
  for(i = 0; i < D; i++){
    for(j = 0; j < D; j++)
      printf("%2X ", state[i][j]);
    printf("\n");
  }
  printf("\n");
}

void PrintState_Column(uint32_t state[D])
{
   int i, j;
  for(i = 0; i < D; i++){
    for(j = 0; j < D; j++)
      printf("%2X ", (state[j]>>(i*S)) & WORDFILTER);
    printf("\n");
  }
  printf("\n");
}

__device__ void AddKey(uint8_t state[D][D], int round)
{
  int i;
  for(i = 0; i < D; i++)
    state[i][0] ^= RC[i][round];
}

__device__ void SCShRMCS(uint8_t state[D][D])
{
  int c,r;
  uint32_t v;
  uint8_t os[D][D];
  memcpy(os, state, D*D); // wklee, optimize this later.
  // for (i = 0; i < D * D; i++) os[i] = state[i];

  for(c = 0; c < D; c++){ // for all columns
    v = 0;
    for(r = 0; r < D; r++) // for all rows in this column i after ShiftRow
      v ^= Table[r*16 + os[r][(r+c)%D]];

    for(r = 1; r <= D; r++){
      state[D-r][c] = (uint8_t)v & WORDFILTER;
      v >>= S;
    }
  }
}

__device__ void SCShRMCS_shared(uint8_t state[D][D])
{
  int c,r, tid = threadIdx.x;
  uint32_t v;
  uint8_t os[D][D];
  memcpy(os, state, D*D); // wklee, optimize this later.
  // for (i = 0; i < D * D; i++) os[i] = state[i];
  __shared__ uint32_t s_table[128];

  if(tid < 128) s_table[tid] = Table[tid];
  __syncthreads();

  // for(c = 0; c < D; c++){ // for all columns
  //   v = 0;
  //   for(r = 0; r < D; r++){ // for all rows in this column i after ShiftRow
  //     v ^= s_table[r*16 + os[r][(r+c)%D]];
  //     // if(tid==0) printf("%u\n", r*16 + os[r][(r+c)%D]); 
  //   }
  //   for(r = 1; r <= D; r++){
  //     state[D-r][c] = (uint8_t)v & WORDFILTER;
  //     v >>= S;
  //   }
  // }
// unrolled r
// #pragma unrolled
  for(c = 0; c < D; c++){ // for all columns
    v = 0;
    v ^= s_table[os[0][(0+c)%D]];     
    v ^= s_table[1*16 + os[1][(1+c)%D]];
    v ^= s_table[2*16 + os[2][(2+c)%D]];
    v ^= s_table[3*16 + os[3][(3+c)%D]];
    v ^= s_table[4*16 + os[4][(4+c)%D]];     
    v ^= s_table[5*16 + os[5][(5+c)%D]];
    v ^= s_table[6*16 + os[6][(6+c)%D]];
    v ^= s_table[7*16 + os[7][(7+c)%D]];      
    state[D-1][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-2][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-3][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-4][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-5][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-6][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-7][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-8][c] = (uint8_t)v & WORDFILTER;      v >>= S;
  }
}

// wklee, can we replace sharedmem with warp shuffle?
__device__ void SCShRMCS_shfl(uint8_t state[D][D])
{
  int c,r, tid = threadIdx.x;
  uint32_t v, tb0, tb1, tb2, tb3;
  uint8_t os[D][D];
  memcpy(os, state, D*D); // wklee, optimize this later.
  // for (i = 0; i < D * D; i++) os[i] = state[i];

  tb0 = Table[tid%32];	tb1 = Table[tid%32 + 32];
  tb2 = Table[tid%32 + 64];	tb3 = Table[tid%32 + 96];  

// unrolled r
// #pragma unrolled
  for(c = 0; c < D; c++){ // for all columns
    v = 0;
    v ^= __shfl_sync(0xffffffff, tb0, os[0][(0+c)%D]);     
    v ^= __shfl_sync(0xffffffff, tb0, 16 + os[1][(1+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb1, os[2][(2+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb1, 16 + os[3][(3+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb2, os[4][(4+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb2, 16 + os[5][(5+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb3, os[6][(6+c)%D]);  
    v ^= __shfl_sync(0xffffffff, tb3, 16 + os[7][(7+c)%D]);  
    
    state[D-1][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-2][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-3][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-4][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-5][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-6][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-7][c] = (uint8_t)v & WORDFILTER;      v >>= S;
    state[D-8][c] = (uint8_t)v & WORDFILTER;      v >>= S;
  }
}

__device__ void Permutation_gpu(uint8_t state[D][D], int R)
{
  int i;
  for(i = 0; i < R; i++) {
    AddKey(state, i);     
#ifdef SHFL        
    SCShRMCS_shfl(state);
#else
    SCShRMCS_shared(state);      
#endif
  }
}

__device__ void PHOTON_Permutation_gpu(unsigned char *State_in)
{
    uint8_t state[D][D];
    int i;

  for (i = 0; i < D * D; i++)
  {
    state[i / D][i % D] = (State_in[i / 2] >> (4 * (i & 1))) & 0xf;
  }
   // PrintState(state);
  Permutation_gpu(state, ROUND);
  // if(threadIdx.x==0) PrintState(state);
  
  for (i = 0; i < D * D/2; i++) State_in[i] = 0;
  for (i = 0; i < D * D; i++)
  {
    State_in[i / 2] |= (state[i / D][i % D] & 0xf) << (4 * (i & 1));
  }
}

/* Definition of basic internal functions */
__device__ uint8_t selectConst(const bool condition, const uint8_t option1, const uint8_t option2)
{
  if (condition) return option1;
  return option2;
}

__device__ void XOR_const_gpu(uint8_t *State_inout,  const uint8_t  Constant)
{
  State_inout[STATE_INBYTES - 1] ^= (Constant << LAST_THREE_BITS_OFFSET);
}

__device__ void XOR_gpu(uint8_t *out,  const uint8_t *in_left,  const uint8_t *in_right, const size_t iolen_inbytes)
{
  size_t i;
  for (i = 0; i < iolen_inbytes; i++) out[i] = in_left[i] ^ in_right[i];
}

__device__  void HASH(
  uint8_t *State_inout,
  const uint8_t *Data_in,
  const uint64_t Dlen_inbytes,
  const uint8_t  Constant)
{
  uint8_t *State = State_inout;
  size_t Dlen_inblocks = (Dlen_inbytes + RATE_INBYTES - 1) / RATE_INBYTES;
  size_t LastDBlocklen;
  size_t i;

  for (i = 0; i < Dlen_inblocks - 1; i++)
  {
    PHOTON_Permutation_gpu(State);
    XOR_gpu(State, State, Data_in + i * RATE_INBYTES, RATE_INBYTES);
  }
  PHOTON_Permutation_gpu(State);  
  LastDBlocklen = Dlen_inbytes - i * RATE_INBYTES;
  XOR_gpu(State, State, Data_in + i * RATE_INBYTES, LastDBlocklen);
  if (LastDBlocklen < RATE_INBYTES) State[LastDBlocklen] ^= 0x01; // ozs

  XOR_const_gpu(State, Constant);
}

__device__  void TAG(uint8_t *Tag_out, uint8_t *State)
{
  // size_t i;
  int j;
  // i = TAG_INBYTES;
  // while (i > SQUEEZE_RATE_INBYTES)
  // {
  //   // printf("ohoh\n");
  //  PHOTON_Permutation_gpu(State);
  //  for(j=0; j <SQUEEZE_RATE_INBYTES; j++) Tag_out[j] = State[j];
  //  Tag_out += SQUEEZE_RATE_INBYTES;
  //  i -= SQUEEZE_RATE_INBYTES;
  // }
  // PHOTON_Permutation_gpu(State); 
  // for(j=0; j <i; j++) Tag_out[j] = State[j];

    PHOTON_Permutation_gpu(State);
    for(j=0; j <SQUEEZE_RATE_INBYTES; j++) Tag_out[j] = State[j];      
    Tag_out += SQUEEZE_RATE_INBYTES;      
    PHOTON_Permutation_gpu(State);
    for(j=0; j <SQUEEZE_RATE_INBYTES; j++) Tag_out[j] = State[j];
}



__global__ void crypto_hash_gpu(uint8_t *out,  uint8_t *in,  uint32_t inlen)
{
    int i, tid = threadIdx.x, bid = blockIdx.x;
    uint8_t State[STATE_INBYTES] = { 0 };
    // uint8_t State[BATCH* STATE_INBYTES] = { 0 };
    uint8_t c0;
    if (inlen == 0)
    {
      // XOR_const(State, 1);
    }
    else if (inlen <= INITIAL_RATE_INBYTES)
    {
      c0 = selectConst((inlen < INITIAL_RATE_INBYTES), 1, 2);      
      for(i=0; i <inlen; i++) State[i] = in[bid*blockDim.x*MLEN + tid*MLEN + i];
      if (inlen < INITIAL_RATE_INBYTES) State[inlen] ^= 0x01; // ozs
      XOR_const_gpu(State, c0);
    }
    else
    {
      // printf("\n multiple blocks\n");
      for(i=0; i <INITIAL_RATE_INBYTES; i++) State[i] = in[bid*blockDim.x*MLEN + tid*MLEN + i];
      inlen -= INITIAL_RATE_INBYTES;
      c0 = selectConst((inlen % RATE_INBYTES) == 0, 1, 2);
      HASH(State, in + INITIAL_RATE_INBYTES, inlen, c0);
    }
    TAG(out + bid*blockDim.x*CRYPTO_BYTES + tid*CRYPTO_BYTES , State);
}



void init_buffer(uint8_t *buffer, uint32_t numbytes)
{
  int i, k;
  for (uint32_t i = 0; i < MAX_MESSAGE_LENGTH; i++) buffer[i] = 0;
  for(k=0; k<BATCH; k++) for (i = 0; i < numbytes; i++)
    // buffer[k*j*numbytes + j*numbytes + i] = rand()%256;
    buffer[k*numbytes + i] = (uint8_t)k + i;
    // buffer[k*numbytes + i] = (uint8_t)i;
}


int main(int argc, char* argv[]) {   
  uint8_t *h_msg, *h_digest, *d_msg, *d_digest, *h_state, *d_state;
  int i, j, k, mlen = MLEN, blocks, threads;
  cudaEvent_t start, stop;
  float elapsed;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaMalloc((void**)&d_digest, BATCH*CRYPTO_BYTES*sizeof(uint8_t));
  cudaMalloc((void**)&d_state, BATCH * D*D * sizeof(uint8_t));
  cudaMalloc((void**)&d_msg, (uint64_t) BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t)); 
   
  cudaMallocHost((void**)&h_digest, BATCH * CRYPTO_BYTES * sizeof(uint8_t));
  cudaMallocHost((void**)&h_msg, (uint64_t) BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t));
  cudaMallocHost((void**)&h_state, BATCH * D*D * sizeof(uint8_t));
    
   // Configure cache
   cudaFuncSetCacheConfig(crypto_hash_gpu, cudaFuncCachePreferShared);
   cudaFuncSetCacheConfig(crypto_hash_gpu_table, cudaFuncCachePreferShared);
   cudaFuncSetCacheConfig(crypto_hash_gpu_bitslice32, cudaFuncCachePreferL1);
   
   threads = BLOCK_SIZE; 
   blocks = BATCH / BLOCK_SIZE;
   if(blocks == 0) blocks = 1;  // wklee, at least one block.
  printf("GPU photon-beetle: using %u blocks and %u threads\n", blocks, threads);

   /* initialize random seed: */   
  // srand (time(NULL));  // comment out this to yield a static poly elements.

  init_buffer(h_msg, mlen);

  // for(k=0; k<2; k++) {printf("\nbatch %u: ", k); for (i = 0; i <mlen; i++) printf("%x ", h_msg[k*mlen + i]);  }
  // for (i = 0; i <mlen; i++) h_digest[i] = 0;

  cudaMemcpy(d_msg, h_msg, (uint64_t)BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t), cudaMemcpyHostToDevice);

  cudaEventRecord(start); 
    
  printf("\n Timing photon-beetle...MLEN: %u\n", MLEN);      

#ifndef BITSLICE
  // Single table-based implementation
  crypto_hash_gpu<<<blocks, threads>>>(d_digest, d_msg, mlen);
  // Double table-based implementation (table2)
  // crypto_hash_gpu_table<<<blocks, threads>>>(d_digest, d_msg, mlen);  
#else
#ifdef   BITSLICE64
  crypto_hash_gpu_bitslice64<<<blocks, threads>>>(d_digest, d_msg, mlen);
#else
  crypto_hash_gpu_bitslice32<<<blocks, threads>>>(d_digest, d_msg, mlen);
#endif  
#endif  
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);   
  cudaEventElapsedTime(&elapsed, start, stop); 
  cudaMemcpy(h_digest, d_digest, BATCH * CRYPTO_BYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
#ifdef PERF
  printf("Latency (ms)\t Average (us) \t Hash/s \t TP (Gbps)\n" );
  printf("%.4f \t %.4f \t %.0f \t %.0f\n", elapsed, elapsed*1000/(BATCH), BATCH/elapsed, (double) 8*BATCH*MLEN/1024/1024/elapsed);     
#endif    
  
#ifdef DEBUG
  for(k=0; k<4; k++) {printf("\n batch %u\n", k);for (i = 0; i <CRYPTO_BYTES; i++) {printf("%x ", h_digest[k*CRYPTO_BYTES + i]);}}
#endif    

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaDeviceReset();
  return 0;
}


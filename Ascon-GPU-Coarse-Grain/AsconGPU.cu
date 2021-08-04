#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <time.h>  
#include <sys/time.h>
#include "params.h"
#include "permutations.h"
// #include "printstate.cuh"
#include "word.h"


__global__ void crypto_hash_gpu_ref(uint8_t* out, uint8_t* in ,
                uint32_t len) {
  /* initialize */
  uint32_t tid = threadIdx.x, bid = blockIdx.x; 
  uint32_t idx_in = bid*BLOCK_SIZE*MLEN + tid*MLEN;
  uint32_t idx_out = bid*BLOCK_SIZE*CRYPTO_BYTES + tid*CRYPTO_BYTES;
  state_t s;
  s.x0 = ASCON_HASH_IV;
  s.x1 = 0;
  s.x2 = 0;
  s.x3 = 0;
  s.x4 = 0;
  P12(&s);

  /* absorb full plaintext blocks */
  while (len >= ASCON_HASH_RATE) {
    s.x0 ^= LOADBYTES(in + idx_in , 8);
    P12(&s);
    in += ASCON_HASH_RATE;
    len -= ASCON_HASH_RATE;
  }
  /* absorb final plaintext block */
  s.x0 ^= LOADBYTES(in + idx_in , len);
  s.x0 ^= PAD(len);
  P12(&s);

  /* squeeze full output blocks */
  len = CRYPTO_BYTES;
  while (len > ASCON_HASH_RATE) {
    STOREBYTES(out + idx_out, s.x0, 8);
    P12(&s);
    out += ASCON_HASH_RATE;
    len -= ASCON_HASH_RATE;
  }
  /* squeeze final output block */
  STOREBYTES(out + idx_out, s.x0, len);
  // printstate("squeeze output", &s);
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
  int i, k, mlen = MLEN, blocks, threads;
  cudaEvent_t start, stop;
  float elapsed;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaMalloc((void**)&d_digest, BATCH*CRYPTO_BYTES*sizeof(uint8_t));
  cudaMalloc((void**)&d_state, (uint64_t) BATCH * STATE_SIZE * sizeof(uint8_t));
  cudaMalloc((void**)&d_msg, (uint64_t) BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t)); 
   
  cudaMallocHost((void**)&h_digest, BATCH * CRYPTO_BYTES * sizeof(uint8_t));
  cudaMallocHost((void**)&h_msg, (uint64_t) BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t));
  cudaMallocHost((void**)&h_state, (uint64_t) BATCH * STATE_SIZE * sizeof(uint8_t));
    
//    // Configure cache
//    cudaFuncSetCacheConfig(crypto_hash_gpu, cudaFuncCachePreferShared);
   
   threads = BLOCK_SIZE; 
   blocks = BATCH / BLOCK_SIZE;
   if(blocks == 0) blocks = 1;  // wklee, at least one block.
  printf("GPU photon-beetle: using %u blocks and %u threads\n", blocks, threads);

//    /* initialize random seed: */   
//   // srand (time(NULL));  // comment out this to yield a static poly elements.
  init_buffer(h_msg, mlen);
  // for(k=0; k<2; k++) {printf("\nbatch %u: ", k); for (i = 0; i <mlen; i++) printf("%x ", h_msg[k*mlen + i]);  }

  cudaMemcpy(d_msg, h_msg, (uint64_t)BATCH * MAX_MESSAGE_LENGTH * sizeof(uint8_t), cudaMemcpyHostToDevice);
  printf("\n Timing ASCON...MLEN: %u\n", MLEN);    
  cudaEventRecord(start);         
  crypto_hash_gpu_ref<<<blocks, threads>>>(d_digest, d_msg, mlen);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);   
  cudaEventElapsedTime(&elapsed, start, stop); 
  cudaMemcpy(h_digest, d_digest, BATCH * CRYPTO_BYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
#ifdef PERF
  printf("Latency (ms)\t Average (us) \t Hash/s \t TP (Gbps)\n" );
  printf("%.4f \t %.4f \t %.0f \t %.0f\n", elapsed, elapsed*1000/(BATCH), BATCH/elapsed, (double) 8*BATCH*MLEN/1024/1024/elapsed);     
#endif    
  
#ifdef DEBUG
  for(k=0; k<4; k++) {printf("\n batch %u\n", k);for (i = 0; i <CRYPTO_BYTES; i++) {printf("%x", h_digest[k*CRYPTO_BYTES + i]);}}
#endif    

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaDeviceReset();
  return 0;
}


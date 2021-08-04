#ifndef PERMUTATIONS_H_
#define PERMUTATIONS_H_

#include <stdint.h>

#include "params.h"
#include "round.h"

#define ASCON_128_KEYBYTES 16
#define ASCON_128A_KEYBYTES 16
#define ASCON_80PQ_KEYBYTES 20

#define ASCON_128_RATE 8
#define ASCON_128A_RATE 16
#define ASCON_HASH_RATE 8

#define ASCON_128_PA_ROUNDS 12
#define ASCON_128_PB_ROUNDS 6

#define ASCON_128A_PA_ROUNDS 12
#define ASCON_128A_PB_ROUNDS 8

#define ASCON_HASH_PA_ROUNDS 12
#define ASCON_HASH_PB_ROUNDS 12

#define ASCON_HASHA_PA_ROUNDS 12
#define ASCON_HASHA_PB_ROUNDS 8

#define ASCON_HASH_BYTES 32

#define ASCON_HASH_IV0 WORD_T(0xf9afb5c6a540dbc7ull)
#define ASCON_HASH_IV1 WORD_T(0xbd2493011445a340ull)
#define ASCON_HASH_IV2 WORD_T(0xcb9ba8b5604d4fc8ull)
#define ASCON_HASH_IV3 WORD_T(0x12a4eede94514c98ull)
#define ASCON_HASH_IV4 WORD_T(0x4bca84c06339f398ull)

#define ASCON_HASHA_IV0 WORD_T(0x0108e46d1b16eb02ull)
#define ASCON_HASHA_IV1 WORD_T(0x5b9b8efdd29083f3ull)
#define ASCON_HASHA_IV2 WORD_T(0x7ad665622891ae4aull)
#define ASCON_HASHA_IV3 WORD_T(0x9dc27156ee3bfc7full)
#define ASCON_HASHA_IV4 WORD_T(0xc61d5fa916801633ull)

#define ASCON_XOF_IV0 WORD_T(0xc75782817e351ae6ull)
#define ASCON_XOF_IV1 WORD_T(0x70045f441d238220ull)
#define ASCON_XOF_IV2 WORD_T(0x5dd5ab52a13e3f04ull)
#define ASCON_XOF_IV3 WORD_T(0x3e378142c30c1db2ull)
#define ASCON_XOF_IV4 WORD_T(0x3735189db624d656ull)

#define ASCON_XOFA_IV0 WORD_T(0x0846d7a5a4b87d44ull)
#define ASCON_XOFA_IV1 WORD_T(0xaa6f1005b3a2dbf4ull)
#define ASCON_XOFA_IV2 WORD_T(0xdc451146f713e811ull)
#define ASCON_XOFA_IV3 WORD_T(0x468cb2532839e30dull)
#define ASCON_XOFA_IV4 WORD_T(0xeb2d429709e96977ull)

// ref
#define ASCON_128_IV                            \
  (((uint64_t)(ASCON_128_KEYBYTES * 8) << 56) | \
   ((uint64_t)(ASCON_128_RATE * 8) << 48) |     \
   ((uint64_t)(ASCON_128_PA_ROUNDS) << 40) |    \
   ((uint64_t)(ASCON_128_PB_ROUNDS) << 32))

#define ASCON_128A_IV                            \
  (((uint64_t)(ASCON_128A_KEYBYTES * 8) << 56) | \
   ((uint64_t)(ASCON_128A_RATE * 8) << 48) |     \
   ((uint64_t)(ASCON_128A_PA_ROUNDS) << 40) |    \
   ((uint64_t)(ASCON_128A_PB_ROUNDS) << 32))

#define ASCON_80PQ_IV                            \
  (((uint64_t)(ASCON_80PQ_KEYBYTES * 8) << 56) | \
   ((uint64_t)(ASCON_128_RATE * 8) << 48) |      \
   ((uint64_t)(ASCON_128_PA_ROUNDS) << 40) |     \
   ((uint64_t)(ASCON_128_PB_ROUNDS) << 32))

#define ASCON_HASH_IV                                                \
  (((uint64_t)(ASCON_HASH_RATE * 8) << 48) |                         \
   ((uint64_t)(ASCON_HASH_PA_ROUNDS) << 40) |                        \
   ((uint64_t)(ASCON_HASH_PA_ROUNDS - ASCON_HASH_PB_ROUNDS) << 32) | \
   ((uint64_t)(ASCON_HASH_BYTES * 8) << 0))

#define ASCON_HASHA_IV                                                 \
  (((uint64_t)(ASCON_HASH_RATE * 8) << 48) |                           \
   ((uint64_t)(ASCON_HASHA_PA_ROUNDS) << 40) |                         \
   ((uint64_t)(ASCON_HASHA_PA_ROUNDS - ASCON_HASHA_PB_ROUNDS) << 32) | \
   ((uint64_t)(ASCON_HASH_BYTES * 8) << 0))

#define ASCON_XOF_IV                          \
  (((uint64_t)(ASCON_HASH_RATE * 8) << 48) |  \
   ((uint64_t)(ASCON_HASH_PA_ROUNDS) << 40) | \
   ((uint64_t)(ASCON_HASH_PA_ROUNDS - ASCON_HASH_PB_ROUNDS) << 32))

#define ASCON_XOFA_IV                          \
  (((uint64_t)(ASCON_HASH_RATE * 8) << 48) |   \
   ((uint64_t)(ASCON_HASHA_PA_ROUNDS) << 40) | \
   ((uint64_t)(ASCON_HASHA_PA_ROUNDS - ASCON_HASHA_PB_ROUNDS) << 32))

__device__ static inline void P12(state_t* s) {
  ROUND(s, 0xf0);
  ROUND(s, 0xe1);
  ROUND(s, 0xd2);
  ROUND(s, 0xc3);
  ROUND(s, 0xb4);
  ROUND(s, 0xa5);
  ROUND(s, 0x96);
  ROUND(s, 0x87);
  ROUND(s, 0x78);
  ROUND(s, 0x69);
  ROUND(s, 0x5a);
  ROUND(s, 0x4b);
}

__device__ static inline void P8(state_t* s) {
  ROUND(s, 0xb4);
  ROUND(s, 0xa5);
  ROUND(s, 0x96);
  ROUND(s, 0x87);
  ROUND(s, 0x78);
  ROUND(s, 0x69);
  ROUND(s, 0x5a);
  ROUND(s, 0x4b);
}

__device__ static inline void P6(state_t* s) {
  ROUND(s, 0x96);
  ROUND(s, 0x87);
  ROUND(s, 0x78);
  ROUND(s, 0x69);
  ROUND(s, 0x5a);
  ROUND(s, 0x4b);
}

#endif /* PERMUTATIONS_H_ */

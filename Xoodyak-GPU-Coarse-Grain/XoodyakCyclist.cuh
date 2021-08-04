#ifndef _XoodyakCyclist_h_
#define _XoodyakCyclist_h_

#include <stdint.h> // uint8_t etc.
#include <stddef.h> // size_t

// #include "Xoodoo-SnP.h"
// #include "Xoodoo.h"
// TODO: constancts for C_D/C_U?

#define Xoodyak_rateHash      16
#define Xoodyak_rateKin       44
#define Xoodyak_rateKout      24
#define Xoodyak_lengthRatchet 16
#define Xoodyak_tagLength     16
#define Xoodoo_implementation      "32-bit reference implementation"
#define Xoodoo_stateSizeInBytes    (3*4*4)
#define Xoodoo_stateAlignment      4
#define Xoodoo_HasNround

#define MAXROUNDS   12
#define NROWS       3
#define NCOLUMS     4
#define NLANES      (NCOLUMS*NROWS)

/*    Round constants    */
#define _rc12   0x00000058
#define _rc11   0x00000038
#define _rc10   0x000003C0
#define _rc9    0x000000D0
#define _rc8    0x00000120
#define _rc7    0x00000014
#define _rc6    0x00000060
#define _rc5    0x0000002C
#define _rc4    0x00000380
#define _rc3    0x000000F0
#define _rc2    0x000001A0
#define _rc1    0x00000012

__constant__ static const uint32_t    RC[MAXROUNDS] = {
    0x00000058, 0x00000038, 0x000003C0,0x000000D0, 0x00000120, 0x00000014, 0x00000060, 0x0000002C, 0x00000380, 0x000000F0, 0x000001A0, 0x00000012
};


#if !defined(ROTL32)
    #if defined (__arm__) && !defined(__GNUC__)
        #define ROTL32(a, offset)                       __ror(a, (32-(offset))%32)
    #elif defined(_MSC_VER)
        #define ROTL32(a, offset)                       _rotl(a, (offset)%32)
    #else
        #define ROTL32(a, offset)                       ((((uint32_t)a) << ((offset)%32)) ^ (((uint32_t)a) >> ((32-(offset))%32)))
    #endif
#endif

#if !defined(READ32_UNALIGNED)
    #if defined (__arm__) && !defined(__GNUC__)
        #define READ32_UNALIGNED(argAddress)            (*((const __packed uint32_t*)(argAddress)))
    #elif defined(_MSC_VER)
        #define READ32_UNALIGNED(argAddress)            (*((const uint32_t*)(argAddress)))
    #else
        #define READ32_UNALIGNED(argAddress)            (*((const uint32_t*)(argAddress)))
    #endif
#endif

#if !defined(WRITE32_UNALIGNED)
    #if defined (__arm__) && !defined(__GNUC__)
        #define WRITE32_UNALIGNED(argAddress, argData)  (*((__packed uint32_t*)(argAddress)) = (argData))
    #elif defined(_MSC_VER)
        #define WRITE32_UNALIGNED(argAddress, argData)  (*((uint32_t*)(argAddress)) = (argData))
    #else
        #define WRITE32_UNALIGNED(argAddress, argData)  (*((uint32_t*)(argAddress)) = (argData))
    #endif
#endif

#if !defined(index)
    #define    index(__x,__y)    ((((__y) % NROWS) * NCOLUMS) + ((__x) % NCOLUMS))
#endif

typedef    uint32_t tXoodooLane;


typedef enum {
    XOODYAK_PHASE_UP,
    XOODYAK_PHASE_DOWN,
} Cyclist_Phase;

typedef enum {
    XOODYAK_MODE_HASH,
    XOODYAK_MODE_KEYED,
} Cyclist_Mode;

typedef struct {
    Cyclist_Phase phase;
    Cyclist_Mode mode;
    size_t rate_absorb;
    size_t rate_squeeze;
    uint8_t state[Xoodoo_stateSizeInBytes];
} Cyclist_Instance;


/* ---------------------------------------------------------------- */

__device__ static  void Xoodoo_Initialize(void *state)
{
    memset(state, 0, NLANES*sizeof(tXoodooLane));
}

/* ---------------------------------------------------------------- */

__device__ static  void Xoodoo_AddByte(void *state, unsigned char byte, unsigned int offset)
{
    ////assert(offset < NLANES*sizeof(tXoodooLane));
    ((unsigned char *)state)[offset] ^= byte;
}

/* ---------------------------------------------------------------- */


__device__ static void fromBytesToWords(tXoodooLane *stateAsWords, const unsigned char *state)
{
    unsigned int i, j;

    for(i=0; i<NLANES; i++) {
        stateAsWords[i] = 0;
        for(j=0; j<sizeof(tXoodooLane); j++)
            stateAsWords[i] |= (tXoodooLane)(state[i*sizeof(tXoodooLane)+j]) << (8*j);
    }
}

__device__ static void fromWordsToBytes(unsigned char *state, const tXoodooLane *stateAsWords)
{
    unsigned int i, j;

    for(i=0; i<NLANES; i++)
        for(j=0; j<sizeof(tXoodooLane); j++)
            state[i*sizeof(tXoodooLane)+j] = (stateAsWords[i] >> (8*j)) & 0xFF;
}

__device__ static  void Xoodoo_Round( tXoodooLane * a, tXoodooLane rc )
{
    unsigned int x, y;
    tXoodooLane    b[NLANES];
    tXoodooLane    p[NCOLUMS];
    tXoodooLane    e[NCOLUMS];

    /* Theta: Column Parity Mixer */
    for (x=0; x<NCOLUMS; ++x)
        p[x] = a[index(x,0)] ^ a[index(x,1)] ^ a[index(x,2)];
    for (x=0; x<NCOLUMS; ++x)
        e[x] = ROTL32(p[(x-1)%4], 5) ^ ROTL32(p[(x-1)%4], 14);
    for (x=0; x<NCOLUMS; ++x)
        for (y=0; y<NROWS; ++y)
            a[index(x,y)] ^= e[x];

    /* Rho-west: plane shift */
    for (x=0; x<NCOLUMS; ++x) {
        b[index(x,0)] = a[index(x,0)];
        b[index(x,1)] = a[index(x-1,1)];
        b[index(x,2)] = ROTL32(a[index(x,2)], 11);
    }
    memcpy( a, b, sizeof(b) );
        
    /* Iota: round constant */
    a[0] ^= rc;

    /* Chi: non linear layer */
    for (x=0; x<NCOLUMS; ++x)
        for (y=0; y<NROWS; ++y)
            b[index(x,y)] = a[index(x,y)] ^ (~a[index(x,y+1)] & a[index(x,y+2)]);
    memcpy( a, b, sizeof(b) );

    /* Rho-east: plane shift */
    for (x=0; x<NCOLUMS; ++x) {
        b[index(x,0)] = a[index(x,0)];
        b[index(x,1)] = ROTL32(a[index(x,1)], 1);
        b[index(x,2)] = ROTL32(a[index(x+2,2)], 8);
    }
    memcpy( a, b, sizeof(b) );
}

__device__ static  void Xoodoo_Permute_Nrounds( void * state, uint32_t nr )
{
    tXoodooLane        a[NLANES];
    unsigned int    i;

    fromBytesToWords(a, (const unsigned char *)state);

    for (i = MAXROUNDS - nr; i < MAXROUNDS; ++i ) {
        Xoodoo_Round( a, RC[i] );
    }

    fromWordsToBytes((unsigned char *)state, a);

}

__device__ static  void Xoodoo_Permute_6rounds( uint32_t * state)
{
    Xoodoo_Permute_Nrounds( state, 6 );
}

__device__ static  void Xoodoo_Permute_12rounds( uint32_t * state)
{
    Xoodoo_Permute_Nrounds( state, 12 );
}


// TODO: For now optimised for dword size (32 bit)
__device__ static inline void memxor(uint8_t *dest, uint8_t const *src, size_t len)
{
    // move dest to be DWORD aligned
    while (len > 0 && ((uintptr_t) dest & (uintptr_t) 0x7)) {
        *dest ^= *src;
        dest += 1;
        src += 1;
        len -= 1;
    }

    // handle full DWORDs
    while (len >= 4) {
        *(uint32_t *) dest ^= * (uint32_t *) src;
        dest += 4;
        src += 4;
        len -= 4;
    }

    // remaining bytes
    while (len > 0) {
        *dest ^= *src;
        dest += 1;
        src += 1;
        len -= 1;
    }
}

__device__ static void Up(
        Cyclist_Instance * const instance,
        uint8_t * const Yi, size_t const Yi_len,
        uint8_t const C_U)
{

    instance->phase = XOODYAK_PHASE_UP;

    if (instance->mode != XOODYAK_MODE_HASH) {
        // Xoodoo_AddByte(instance->state, C_U, Xoodoo_stateSizeInBytes - 1);
        ((unsigned char *)instance->state)[Xoodoo_stateSizeInBytes - 1] ^= C_U;
    }
    Xoodoo_Permute_Nrounds(&(instance->state), 12);

    if (Yi != NULL) {
        memcpy(Yi, (unsigned char*)instance->state+0, Yi_len);
    }
}

__device__ static void Down(
        Cyclist_Instance * const instance,
        uint8_t const * const X, size_t X_len,
        uint8_t const C_D)
{

    unsigned int i;
    instance->phase = XOODYAK_PHASE_DOWN;

    for(i=0; i<X_len; i++)
        ((unsigned char *)instance->state)[0+i] ^= X[i];
    ((unsigned char *)instance->state)[X_len] ^= 0x01;
    ((unsigned char *)instance->state)[Xoodoo_stateSizeInBytes - 1] ^= (instance->mode == XOODYAK_MODE_HASH) ? (C_D & 0x01) : C_D;

    // Xoodoo_AddBytes(instance->state, X, 0, X_len);
    // Xoodoo_AddByte(instance->state, 0x01, X_len);
    // Xoodoo_AddByte(instance->state, (instance->mode == XOODYAK_MODE_HASH) ? (C_D & 0x01) : C_D, Xoodoo_stateSizeInBytes - 1);    
}


__device__ static void AbsorbAny(
        Cyclist_Instance * const instance,
        uint8_t const *X, size_t X_len,
        size_t const r, uint8_t const C_D)
{
    //////assert(X != NULL || X_len == 0);

    size_t block_len;

    // First block, possibly empty block
    {
        block_len = X_len < r ? X_len : r;
        if (instance->phase != XOODYAK_PHASE_UP) {
            Up(instance, NULL, 0, 0x00);
        }
        Down(instance, X, block_len, C_D);

        X += block_len;
        X_len -= block_len;
    }

    // Rest of the blocks
    while (X_len != 0) {
        block_len = X_len < r ? X_len : r;

        Up(instance, NULL, 0, 0x00);
        Down(instance, X, block_len, 0x00);

        X += block_len;
        X_len -= block_len;
    }
}


__device__ static void AbsorbKey(
        Cyclist_Instance * const instance,
        uint8_t const * const K, uint8_t K_len,
        uint8_t const * const id, uint8_t id_len,
        uint8_t const * const counter, size_t counter_len)
{
    instance->mode = XOODYAK_MODE_KEYED;
    instance->rate_absorb = Xoodoo_stateSizeInBytes - 4;;

    // Copy concatination of the key, id, and length of id in temp
    uint8_t temp[Xoodyak_rateKin];
    memcpy(temp, K, K_len);
    if (id_len != 0) {
        memcpy(temp + K_len, id, id_len);
    }
    *(temp + K_len + id_len) = id_len;

    // Absorb the concatination
    AbsorbAny(instance, temp, K_len + id_len + 1, instance->rate_absorb, 0x02);

    if (counter_len != 0) {
        AbsorbAny(instance, counter, counter_len, 1, 0x00);
    }
}

__device__ static  inline void Cyclist_Initialize(
        Cyclist_Instance * const instance,
        uint8_t const * const K, uint8_t K_len,
        uint8_t const * const id, uint8_t id_len,
        uint8_t const * const counter, size_t counter_len)
{
    instance->phase = XOODYAK_PHASE_UP;
    instance->mode = XOODYAK_MODE_HASH;
    instance->rate_absorb = Xoodyak_rateHash;
    instance->rate_squeeze = Xoodyak_rateHash;

    // Xoodoo_StaticInitialize(); // wklee, does nothing
    Xoodoo_Initialize(instance->state);

    if (K_len != 0) {
        AbsorbKey(instance, K, K_len, id, id_len, counter, counter_len);
    }
}



__device__ static  void Cyclist_Ratched(Cyclist_Instance *instance);

__device__ static void Crypt(
        Cyclist_Instance *instance,
        uint8_t *O,
        uint8_t const *I, size_t I_len,
        bool Decrypt_Flag)
{
    //assert(I != NULL || I_len == 0);
    //assert(O != NULL || I_len == 0);

    size_t block_len;

    // TODO: move both blocks in one loop? variable for first block C_U?
    // First block, possibly empty block
    {
        block_len = I_len < Xoodyak_rateKout ? I_len : Xoodyak_rateKout;

        Up(instance, O, block_len, 0x80);
        memxor(O, I, block_len);
        Down(instance, Decrypt_Flag ? O : I, block_len, 0x00);

        O += block_len;
        I += block_len;
        I_len -= block_len;
    }

    // Rest of the blocks
    while (I_len != 0) {
        block_len = I_len < Xoodyak_rateKout ? I_len : Xoodyak_rateKout;

        Up(instance, O, block_len, 0x00);
        memxor(O, I, block_len);
        Down(instance, Decrypt_Flag ? O : I, block_len, 0x00);

        O += block_len;
        I += block_len;
        I_len -= block_len;
    }
}

__device__ static void SqueezeAny(
        Cyclist_Instance * const instance,
        uint8_t *Y, size_t l,
        uint8_t const C_U)
{
    //assert(Y != NULL || l == 0);

    size_t block_len;

    // TODO: move both loops in one loop? variable for first block C_U?
    {
        block_len = l < instance->rate_squeeze ? l : instance->rate_squeeze;
        Up(instance, Y, block_len, C_U);

        Y += block_len;
        l -= block_len;
    }

    while (l > 0) {
        Down(instance, NULL, 0, 0x00);

        block_len = l < instance->rate_squeeze ? l : instance->rate_squeeze;
        Up(instance, Y, block_len, 0x00);

        Y += block_len;
        l -= block_len;
    }
}


// =====================================================================
//                           Public interface
// =====================================================================


__device__ void Cyclist_Absorb(
        Cyclist_Instance * const instance,
        uint8_t *X, size_t const X_len)
{
    //assert(X != NULL || X_len == 0);

    AbsorbAny(instance, X, X_len, instance->rate_absorb, 0x03);
}

__device__ void Cyclist_Encrypt(
        Cyclist_Instance * const instance,
        uint8_t * const C,
        uint8_t const * const P, size_t const P_len)
{
    //assert(P != NULL || P_len == 0);
    //assert(C != NULL || P_len == 0);

    //assert(instance->mode == XOODYAK_MODE_KEYED);

    Crypt(instance, C, P, P_len, false);
}

__device__ void Cyclist_Decrypt(
        Cyclist_Instance * const instance,
        uint8_t * const P,
        uint8_t const * const C, size_t const C_len)
{
    //assert(C != NULL || C_len == 0);
    //assert(P != NULL || C_len == 0);

    //assert(instance->mode == XOODYAK_MODE_KEYED);

    Crypt(instance, P, C, C_len, true);
}

__device__ void Cyclist_Squeeze(
        Cyclist_Instance * const instance,
        uint8_t * const Y, size_t const Y_len)
{
    //assert(Y != NULL || Y_len == 0);

    SqueezeAny(instance, Y, Y_len, 0x40);
}

__device__ void Cyclist_SqueezeKey(
        Cyclist_Instance * const instance,
        uint8_t * const K, size_t const K_len)
{
    //assert(K != NULL || K_len == 0);

    //assert(instance->mode == XOODYAK_MODE_KEYED);

    SqueezeAny(instance, K, K_len, 0x20);
}

__device__ void Cyclist_Ratched(Cyclist_Instance * const instance)
{
    //assert(instance->mode == XOODYAK_MODE_KEYED);

    uint8_t temp[Xoodyak_lengthRatchet];
    SqueezeAny(instance, temp, Xoodyak_lengthRatchet, 0x10);
    AbsorbAny(instance, temp, Xoodyak_lengthRatchet, instance->rate_absorb, 0x00);
}


#endif

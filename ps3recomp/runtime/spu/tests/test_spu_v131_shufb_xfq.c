#include <stdio.h>
#include <stdint.h>
#include <string.h>
int g_spu_xfloat = 1;
uint32_t g_spu_xflim[8] = { 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0xFF7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu };
#include "spu_helpers.h"
static uint64_t s=88172645463325252ull; static uint32_t r32(void){s^=s<<13;s^=s>>7;s^=s<<17;return (uint32_t)s;}
int main(){ long bad=0;
  for (long it=0; it<20000000; it++){ u128 a,b,c; for(int i=0;i<4;i++){a._u32[i]=r32();b._u32[i]=r32();c._u32[i]=r32();}
    if (it < 256*16) { for(int k=0;k<16;k++) c._u8[k]=(uint8_t)(it+k*17); }
    u128 x=spu_shufb_ref(a,b,c), y=spu_shufb(a,b,c);
    if (memcmp(&x,&y,16)) { if(bad++<5) printf("shufb mismatch\n"); } }
  /* xfloat off */
  g_spu_xfloat=0; for(int i=0;i<4;i++){g_spu_xflim[i]=0xFFFFFFFFu;g_spu_xflim[4+i]=0x7FFFFFFFu;}
  for (uint64_t u=0; u<=0xFFFFFFFFull; u+=7){ __m128 v=_mm_castsi128_ps(_mm_set1_epi32((int)(uint32_t)u));
    if (_mm_cvtsi128_si32(_mm_castps_si128(spu__xfq(v)))!=(int)(uint32_t)u) { if(bad++<10) printf("xfq-off mismatch %08X\n",(uint32_t)u);} }
  printf("bad=%ld\n",bad); return bad!=0; }

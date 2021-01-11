/*
  Test as:
    $ gcc -ggdb3 -O2 -m32 test-memmove-xmm-unaligned.c -o test-memmove-xmm-unaligned -Wall && ./test-memmove-xmm-unaligned
    $ gcc -ggdb3 -O2 -m64 -mavx test-memmove-xmm-unaligned.c -o test-memmove-xmm-unaligned-64 -Wall && nice -n19 ./test-memmove-xmm-unaligned-64-avx128
    $ gcc -ggdb3 -O2 -m32 -mavx test-memmove-xmm-unaligned.c -o test-memmove-xmm-unaligned-32-avx128 -Wall && nice -n19 ./test-memmove-xmm-unaligned-32-avx128
  Error example:
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset= 8031729; expected=007A8DF1( 8031729) actual=007A8DF3( 8031731) bit_mismatch=00000002; iteration=2
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset=43626993; expected=0299B1F1(43626993) actual=0299B1F3(43626995) bit_mismatch=00000002; iteration=3
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset=25404913; expected=0183A5F1(25404913) actual=0183A5F3(25404915) bit_mismatch=00000002; iteration=4
    ...
*/

#include <string.h> /* memmove */
#include <stdlib.h> /* exit */
#include <stdio.h>  /* fprintf */

#include <sys/mman.h> /* mlock() */
#include <emmintrin.h> /* movdqu, sfence, movntdq */

typedef unsigned int u32;

static void memmove_si128u (__m128i_u * dest, __m128i_u const *src, size_t items) __attribute__((noinline));
static void memmove_si128u (__m128i_u * dest, __m128i_u const *src, size_t items)
{
    // emulate behaviour of optimised block for __memmove_sse2_unaligned:
    // sfence
    // loop(backwards) {
    //   8x movdqu  mem->%xmm{N}
    //   8x movntdq %xmm{N}->mem
    // }
    // source: https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/i386/i686/multiarch/memcpy-sse2-unaligned.S;h=9aa17de99c9c3415a9b5ac28fd9f1eb4457f916d;hb=HEAD#l244

    // ASSUME: if ((unintptr_t)dest > (unintptr_t)src) {
    dest += items - 1;
    src  += items - 1;
    _mm_sfence();
    for (; items != 0; items-=8, dest-=8, src-=8)
    {
        __m128i xmm0 = _mm_loadu_si128(src-0); // movdqu
        __m128i xmm1 = _mm_loadu_si128(src-1); // movdqu
        __m128i xmm2 = _mm_loadu_si128(src-2); // movdqu
        __m128i xmm3 = _mm_loadu_si128(src-3); // movdqu
        __m128i xmm4 = _mm_loadu_si128(src-4); // movdqu
        __m128i xmm5 = _mm_loadu_si128(src-5); // movdqu
        __m128i xmm6 = _mm_loadu_si128(src-6); // movdqu
        __m128i xmm7 = _mm_loadu_si128(src-7); // movdqu
        if (0)
        {
          // this would work:
          _mm_storeu_si128(dest-0, xmm0);// movdqu
          _mm_storeu_si128(dest-1, xmm1);// movdqu
          _mm_storeu_si128(dest-2, xmm2);// movdqu
          _mm_storeu_si128(dest-3, xmm3);// movdqu
          _mm_storeu_si128(dest-4, xmm4);// movdqu
          _mm_storeu_si128(dest-5, xmm5);// movdqu
          _mm_storeu_si128(dest-6, xmm6);// movdqu
          _mm_storeu_si128(dest-7, xmm7);// movdqu
        }
        else
        {
          _mm_stream_si128(dest-0, xmm0); // movntdq
          _mm_stream_si128(dest-1, xmm1); // movntdq
          _mm_stream_si128(dest-2, xmm2); // movntdq
          _mm_stream_si128(dest-3, xmm3); // movntdq
          _mm_stream_si128(dest-4, xmm4); // movntdq
          _mm_stream_si128(dest-5, xmm5); // movntdq
          _mm_stream_si128(dest-6, xmm6); // movntdq
          _mm_stream_si128(dest-7, xmm7); // movntdq
        }
    }
    _mm_sfence();
}

static void do_memmove (u32 * buf, size_t buf_elements, size_t iter) __attribute__((noinline));
static void do_memmove (u32 * buf, size_t buf_elements, size_t iter)
{
  size_t elements_to_move = buf_elements / 2;
  size_t salt = 0x51515151;

  // "memset" buffer with 0, 1, 2, 3, ...
  for (u32 i = 0; i < elements_to_move; i++) buf[i] = i + salt;

  u32 * dst = buf + sizeof (__m128i) / sizeof (u32); // minimal offset: 16 bytes

  // __memmove_sse2_unaligned
  // memmove(dst, buf, elements_to_move * sizeof (u32));
  memmove_si128u((__m128i_u *)dst, (__m128i_u const *)buf, elements_to_move * sizeof (u32) / sizeof (__m128i));

  // validate target buffer buffer with 0, 1, 2, 3, ...
  for (u32 i = 0; i < elements_to_move; i++)
  {
    u32 v = dst[i];
    u32 e = i + salt;
    if (v != e)
      fprintf (stderr,
               "Bad result in memmove(dst=%p, src=%p, len=%zd)"
               ": offset=%8u; expected=%08X(%8u) actual=%08X(%8u) bit_mismatch=%08X; iteration=%zu\n",
               dst, buf, elements_to_move * sizeof (u32),
               i, e, e, v, v, v^e, iter);
  }
}

int main (void)
{
  //size_t size = 8 * 1024 * 1024;
  //size_t size = 8 * 1024 * 1024;
  size_t size = 1 * 1024 * 1024 * 1024;
  void * buf = malloc(size);
  mlock (buf, size);
  // wait for a failure
  for (size_t n = 0; ;++n) {
    do_memmove(buf, size / sizeof (u32), n);
  }
  free(buf);
}

/*
  Test as:
    $ g++ -ggdb3 -O2 -m64 -mavx test-memmove-xmm-unaligned-1.cc -o test-memmove-xmm-unaligned-1-64-avx128 -Wall && nice -n19 ./test-memmove-xmm-unaligned-1-64-avx128
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

#include <vector>

typedef unsigned int u32;

static void memmove_si128u (__m128i_u * dest, __m128i_u const *src, size_t items) __attribute__((noinline));
static void memmove_si128u (__m128i_u * dest, __m128i_u const *src, size_t items)
{
    dest += items - 1;
    src  += items - 1;
    _mm_sfence();
    for (; items != 0; items-=1, dest-=1, src-=1)
    {
        __m128i xmm0 = _mm_loadu_si128(src-0); // movdqu
        if (0)
        {
          // this would work:
          _mm_storeu_si128(dest-0, xmm0);// movdqu
        }
        else
        {
          // this causes single bit memory corruption
          _mm_stream_si128(dest-0, xmm0); // movntdq
        }
    }
    _mm_sfence();
}

static bool seen_error = false;

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
    {
      fprintf (stderr,
               "Bad result in memmove(dst=%p, src=%p, len=%zd)"
               ": offset=%8u; expected=%08X(%8u) actual=%08X(%8u) bit_mismatch=%08X; iteration=%zu\n",
               dst, buf, elements_to_move * sizeof (u32),
               i, e, e, v, v, v^e, iter);
      seen_error = true;
    }
  }
}

static std::vector<void *> ram_stash;

static void take_ram(void)
{
    size_t size = 128 * 1024 * 1024;
    fprintf(stderr, "alloc more: %zu\n", size);
    void * chunk = malloc(size);
    memset(chunk, '!', size);
    ram_stash.push_back(chunk);
}

static void free_ram(void)
{
    if (!ram_stash.empty())
    {
        fprintf(stderr, "freeing all stash\n");
        auto p = ram_stash.begin();
        auto e = ram_stash.end();
        for (; p != e; ++p)
            free (*p);
        ram_stash.clear();
    }
}

int main (void)
{
  for (size_t n = 0; ;++n)
  {
    size_t size = 128 * 1024 * 1024;
    void * buf = malloc(size);
    mlock (buf, size);
    // wait for a failure

    do_memmove((u32 *)buf, size / sizeof (u32), n);
    do_memmove((u32 *)buf, size / sizeof (u32), n);
    do_memmove((u32 *)buf, size / sizeof (u32), n);
    do_memmove((u32 *)buf, size / sizeof (u32), n);

    free(buf);

    if (seen_error)
    {
        if (0) free_ram();
    }
    else if (n % 10 == 0)
        take_ram();
  }
}

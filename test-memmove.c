/*
  Test as:
    $ gcc -ggdb3 -O0 -m32 test-memmove.c -o test-memmove -Wall && ./test-memmove
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

typedef unsigned int u32;

static void do_memmove (u32 * buf, size_t buf_elements, size_t iter) __attribute__((noinline));
static void do_memmove (u32 * buf, size_t buf_elements, size_t iter)
{
  size_t elements_to_move = buf_elements / 2;

  // "memset" buffer with 0, 1, 2, 3, ...
  for (u32 i = 0; i < elements_to_move; i++) buf[i] = i;

  u32 * dst = buf + 33;

  // __memmove_sse2_unaligned
  memmove(dst, buf, elements_to_move * sizeof (u32));

  for (u32 i = 0; i < elements_to_move; i++)
  {
    u32 v = dst[i];
    if (v != i)
      fprintf (stderr,
               "Bad result in memmove(dst=%p, src=%p, len=%zd)"
               ": offset=%8u; expected=%08X(%8u) actual=%08X(%8u) bit_mismatch=%08X; iteration=%zu\n",
               dst, buf, elements_to_move * sizeof (u32),
               i, i, i, v, v, v^i, iter);
  }
}

int main (void)
{
  size_t size = 256 * 1024 * 1024;
  void * buf = malloc(size);
  mlock (buf, size);
  // wait for a failure
  for (size_t n = 0; ;++n) {
    do_memmove(buf, size / sizeof (u32), n);
  }
  free(buf);
}

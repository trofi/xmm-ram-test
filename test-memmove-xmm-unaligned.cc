/*
  Test as:
    $ g++ -ggdb3 -O2 -m64 -mavx test-memmove-xmm-unaligned.cc -o test-memmove-xmm-unaligned-64-avx128 -Wall && nice -n19 ./test-memmove-xmm-unaligned-64-avx128
  Error example:
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset= 8031729; expected=007A8DF1( 8031729) actual=007A8DF3( 8031731) bit_mismatch=00000002; iteration=2
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset=43626993; expected=0299B1F1(43626993) actual=0299B1F3(43626995) bit_mismatch=00000002; iteration=3
    Bad result in memmove(dst=0xd7cf5094, src=0xd7cf5010, len=268435456): offset=25404913; expected=0183A5F1(25404913) actual=0183A5F3(25404915) bit_mismatch=00000002; iteration=4
    ...

  The idea:
    Use a sequence of 'movntdq' (or vmovntdq, both fail) (aka _mm_storeu_si128) stores to implement
    'memmove' and validate the result.
  Note: 'movdq' (_mm_stream_si128) does not cause the error.

  Test operation details:
  - test allocates 128MB chunks and does 400 memmove_si128u() iterations on the last allocated chunk
  - on bad hardware test usually corrupts one bit of RAM (test verifies RAM contents)
  - on machines without RAM problems test silently OOMs
  - on machines with RAM problems test keeps reporting 'Bad result in memmove...' (as above)

  Observed failure details on my machine:
  - over past 7 years I have experienced rare SIGSEGVs in userspace and kernel space
    without meaningful backtraces. It always looked as a memory corruption.
    One day I have noticed glibc memmove() test failures on my machine. That is never
    supposed to happen. a few more details: https://lkml.org/lkml/2018/6/16/120
  - Failures always started happening when OS allocated around 18GB of data (I guess
    started using third DIMM module).
  - Once I've reshuffled DIMM modules in motherboard this test started crashing at
    2GB mark. That's when I've started strongly suspecting hardware bug and not a
    software bug.
  - After isolating bad DIMM I've unplugged bad DIMM module from motherboard and
    plugged new ones. SIGSEGVs disappeared and this test stopped failing.
  - memtest86+ did not detect any errors in any DIMM configurations on motherboard.

  Speculations:
  - I think it's an unusual case of memory corruption somewhere on the boundary
    of CPU memory controller and DRAM controllers.
  - 'movntdq' differs from 'movdq' in a few regards:
    * write combines happen on CPU side(?) in a weakly ordered fashion (as opposed
      to strong ordering).
    * data being written in evicted from the cache and has better chance to be observed
      as corrupted after reloading into cache.
  - memtest86+ might not be able to detect the problem because non-temporal (NT)
    instructions are not used in tests. Wide (128-bit) writes are not issued and don't
    trigger such write striping on dual-channel setup.

  Hardware details:
  - ~7 years old desktop PC
  - Base Board Information
        Manufacturer: Gigabyte Technology Co., Ltd.
        Product Name: H77M-D3H
  - 4x8GB DDR3 QUM3U-8G1600C11R RAM chips witking at 1333 MT/s

    Memory Device
        Array Handle: 0x0007
        Error Information Handle: Not Provided
        Total Width: 64 bits
        Data Width: 64 bits
        Size: 8192 MB
        Form Factor: DIMM
        Set: None
        Locator: ChannelA-DIMM0
        Bank Locator: BANK 0
        Type: DDR3
        Type Detail: Synchronous
        Speed: 1333 MT/s
        Manufacturer: 0000
        Serial Number: 00000000
        Asset Tag: 9876543210
        Part Number: QUM3U-8G1600C11R
        Rank: 2
        Configured Clock Speed: 1333 MT/s

  - CPU
    Processor Information
        Socket Designation: Intel(R) Core(TM) i7-2700K CPU @ 3.50GHz
        Type: Central Processor
        Family: Core i7
        Manufacturer: Intel
        ID: A7 06 02 00 FF FB EB BF
        Signature: Type 0, Family 6, Model 42, Stepping 7
        Flags:
                FPU (Floating-point unit on-chip)
                VME (Virtual mode extension)
                DE (Debugging extension)
                PSE (Page size extension)
                TSC (Time stamp counter)
                MSR (Model specific registers)
                PAE (Physical address extension)
                MCE (Machine check exception)
                CX8 (CMPXCHG8 instruction supported)
                APIC (On-chip APIC hardware supported)
                SEP (Fast system call)
                MTRR (Memory type range registers)
                PGE (Page global enable)
                MCA (Machine check architecture)
                CMOV (Conditional move instruction supported)
                PAT (Page attribute table)
                PSE-36 (36-bit page size extension)
                CLFSH (CLFLUSH instruction supported)
                DS (Debug store)
                ACPI (ACPI supported)
                MMX (MMX technology supported)
                FXSR (FXSAVE and FXSTOR instructions supported)
                SSE (Streaming SIMD extensions)
                SSE2 (Streaming SIMD extensions 2)
                SS (Self-snoop)
                HTT (Multi-threading)
                TM (Thermal monitor supported)
                PBE (Pending break enabled)
        Version: Intel(R) Core(TM) i7-2700K CPU @ 3.50GHz
        Voltage: 1.2 V
        External Clock: 100 MHz
        Max Speed: 7000 MHz
        Current Speed: 3600 MHz
        Status: Populated, Enabled
        Upgrade: Other
        L1 Cache Handle: 0x0004
        L2 Cache Handle: 0x0005
        L3 Cache Handle: 0x0006
        Serial Number: Not Specified
        Asset Tag: Fill By OEM
        Part Number: Fill By OEM
        Core Count: 4
        Core Enabled: 1
        Thread Count: 2
        Characteristics:
                64-bit capable
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
    // emulate behaviour of optimised block for __memmove_sse2_unaligned:
    // sfence
    // loop(backwards) {
    //   8x movdqu  mem->%xmm{N}
    //   8x movntdq %xmm{N}->mem
    // }
    // source: https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/i386/i686/multiarch/memcpy-sse2-unaligned.S;h=9aa17de99c9c3415a9b5ac28fd9f1eb4457f916d;hb=HEAD#l244

    // ASSUME: if ((unintptr_t)dest > (unintptr_t)src) {

    /* Code generated by gcc-8 in case you need to get something similar or just hardcode it:

    Dump of assembler code for function memmove_si128u(__m128i_u*, __m128i_u const*, size_t):
       0x0000000000000ae0 <+0>:     sfence 
       0x0000000000000ae3 <+3>:     mov    %rdx,%rax
       0x0000000000000ae6 <+6>:     shl    $0x4,%rax
       0x0000000000000aea <+10>:    sub    $0x10,%rax
       0x0000000000000aee <+14>:    add    %rax,%rdi
       0x0000000000000af1 <+17>:    add    %rax,%rsi
       0x0000000000000af4 <+20>:    test   %rdx,%rdx
       0x0000000000000af7 <+23>:    je     0xb64 <memmove_si128u(__m128i_u*, __m128i_u const*, size_t)+132>
       0x0000000000000af9 <+25>:    nopl   0x0(%rax)
       0x0000000000000b00 <+32>:    vmovdqu -0x10(%rsi),%xmm6
       0x0000000000000b05 <+37>:    vmovdqu -0x20(%rsi),%xmm5
       0x0000000000000b0a <+42>:    add    $0xffffffffffffff80,%rdi
       0x0000000000000b0e <+46>:    add    $0xffffffffffffff80,%rsi
       0x0000000000000b12 <+50>:    vmovdqu 0x50(%rsi),%xmm4
       0x0000000000000b17 <+55>:    vmovdqu 0x40(%rsi),%xmm3
       0x0000000000000b1c <+60>:    vmovdqu 0x30(%rsi),%xmm2
       0x0000000000000b21 <+65>:    vmovdqu 0x20(%rsi),%xmm1
       0x0000000000000b26 <+70>:    vmovdqu 0x10(%rsi),%xmm0
       0x0000000000000b2b <+75>:    vmovdqu 0x80(%rsi),%xmm7
       0x0000000000000b33 <+83>:    vmovntdq %xmm6,0x70(%rdi)
       0x0000000000000b38 <+88>:    vmovntdq %xmm5,0x60(%rdi)
       0x0000000000000b3d <+93>:    vmovntdq %xmm4,0x50(%rdi)
       0x0000000000000b42 <+98>:    vmovntdq %xmm3,0x40(%rdi)
       0x0000000000000b47 <+103>:   vmovntdq %xmm2,0x30(%rdi)
       0x0000000000000b4c <+108>:   vmovntdq %xmm1,0x20(%rdi)
       0x0000000000000b51 <+113>:   vmovntdq %xmm7,0x80(%rdi)
       0x0000000000000b59 <+121>:   vmovntdq %xmm0,0x10(%rdi)
       0x0000000000000b5e <+126>:   sub    $0x8,%rdx
       0x0000000000000b62 <+130>:   jne    0xb00 <memmove_si128u(__m128i_u*, __m128i_u const*, size_t)+32>
       0x0000000000000b64 <+132>:   sfence 
       0x0000000000000b67 <+135>:   retq
     */
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
          // this causes single bit memory corruption
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

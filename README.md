These are RAM tests mentioned in http://trofi.github.io/posts/209-tracking-down-mysterious-memory-corruption.html.

Source files contain example build command and output in case of found unexpected bit flips.

`test-memmove-xmm-unaligned.cc` is probably most complete and most commented final result.

WARNINGs:
- Tests don't handle errors nicely (like `mlock()` calls) to allow being ran as root and as user.
- Tests intentionally try to eat all you RAM chunk by chunk. Make sure you have read the source
  before running it.

Good luck!

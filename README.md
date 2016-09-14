 Fastest implementation of the fully scale-
 and rotation-invariant LATCH 512-bit binary
 feature descriptor as described in the 2015
 paper by Levi and Hassner:

 "LATCH: Learned Arrangements of Three Patch Codes"
 http://arxiv.org/abs/1501.03719

 See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:

 "The CUDA LATCH Binary Descriptor"
 http://arxiv.org/abs/1609.03986

 And the original LATCH project's website:
 http://www.openu.ac.il/home/hassner/projects/LATCH/

 See my GitHub for the CUDA version, which is extremely fast.

 My implementation uses multithreading, SSE2/3, and 
 many many careful optimizations to implement the
 algorithm as described in the paper, but at great speed.
 This implementation outperforms the reference implementation by 600%
 single-threaded or 2200% multi-threaded (!) while exactly matching
 the reference implementation's output and capabilities.

 All functionality is contained in the file LATCH.h.
 'main.cpp' is simply a sample test harness with example usage and
 performance testing.

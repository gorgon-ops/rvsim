# cache.s -- locality, capacity and conflict, one after the other.
#
# Watch the L1D fill and thrash:
#   python -m rvsim run programs/cache.s --l1d-assoc 1
#   python -m rvsim run programs/cache.s --l1d-assoc 4

    .equ HALT, 0x10000004

    .text
_start:
    la   s0, buffer

    # ---- 1. sequential walk: one miss per line, then free hits --------
    # With 32-byte lines, one miss pulls in eight words. The seven that
    # follow are hits, so the miss rate should settle near 1/8.
    li   t0, 0
    li   t1, 64
seq:
    slli t2, t0, 2
    add  t3, s0, t2
    lw   t4, 0(t3)
    add  s1, s1, t4
    addi t0, t0, 1
    blt  t0, t1, seq

    # ---- 2. tight reuse: the working set fits, so everything hits -----
    li   t0, 0
    li   t1, 100
reuse:
    lw   t4, 0(s0)
    lw   t5, 4(s0)
    lw   t6, 8(s0)
    add  s2, s2, t4
    addi t0, t0, 1
    blt  t0, t1, reuse

    # ---- 3. conflict: three addresses that map to the same set --------
    # The stride has to be a multiple of (sets x line size) to collide,
    # and that product depends on the associativity: a 1 KB cache with
    # 32-byte lines has 32 sets when direct-mapped but only 8 when 4-way.
    # 1024 bytes is a multiple of both, so these three addresses land in
    # one set in every configuration. Direct-mapped can hold one of them
    # and thrashes on every access; 4-way holds all three and never misses
    # after the first pass.
    li   t0, 0
    li   t1, 150
conflict:
    lw   t4, 0(s0)
    lw   t5, 1024(s0)
    lw   t6, 2048(s0)
    add  s3, s3, t4
    addi t0, t0, 1
    blt  t0, t1, conflict

    # ---- 4. strided walk: one miss per access, every access -----------
    # A 64-byte stride skips a whole line each time, so the eight-word
    # line the machine just fetched is never reused.
    li   t0, 0
    li   t1, 24
stride:
    slli t2, t0, 6
    add  t3, s0, t2
    lw   t4, 0(t3)
    add  s4, s4, t4
    addi t0, t0, 1
    blt  t0, t1, stride

    # ---- 5. dirty lines: writes now have to be written back -----------
    li   t0, 0
    li   t1, 32
dirty:
    slli t2, t0, 5
    add  t3, s0, t2
    sw   t0, 0(t3)
    addi t0, t0, 1
    blt  t0, t1, dirty

    j    _halt

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
buffer:
    .word 1, 2, 3, 4, 5, 6, 7, 8
    .word 9, 10, 11, 12, 13, 14, 15, 16
    .space 3040

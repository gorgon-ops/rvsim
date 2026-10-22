# sort.s -- a real algorithm, so the numbers mean something.
#
#   python -m rvsim run programs/sort.s --dump-memory
#
# Insertion sort over 16 words, then a linear search and a checksum.
# Data-dependent branches, a load-use hazard in the inner loop and a
# working set that fits comfortably in L1: a realistic mix rather than a
# microbenchmark.

    .equ HALT, 0x10000004
    .equ N,    16

    .text
_start:
    la   s0, array
    li   s1, N

# ---- insertion sort ---------------------------------------------------
# for (i = 1; i < N; i++) { key = a[i]; j = i-1;
#   while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
#   a[j+1] = key; }
    li   t0, 1                   # i
outer:
    bge  t0, s1, sorted

    slli t1, t0, 2
    add  t1, s0, t1
    lw   t2, 0(t1)               # key = a[i]
    addi t3, t0, -1              # j = i - 1

inner:
    bltz t3, place
    slli t4, t3, 2
    add  t4, s0, t4
    lw   t5, 0(t4)               # a[j]  <- load-use hazard with the bge
    ble  t5, t2, place
    sw   t5, 4(t4)               # a[j+1] = a[j]
    addi t3, t3, -1
    j    inner

place:
    addi t4, t3, 1
    slli t4, t4, 2
    add  t4, s0, t4
    sw   t2, 0(t4)               # a[j+1] = key

    addi t0, t0, 1
    j    outer

sorted:
# ---- checksum, weighted by position so order matters ------------------
    li   t0, 0                   # i
    li   s2, 0                   # checksum
sum:
    bge  t0, s1, searched
    slli t1, t0, 2
    add  t1, s0, t1
    lw   t2, 0(t1)
    addi t3, t0, 1
    mul  t2, t2, t3
    add  s2, s2, t2
    addi t0, t0, 1
    j    sum

searched:
# ---- linear search for a value that is present ------------------------
    li   t6, 42
    li   t0, 0
    li   s3, -1
find:
    bge  t0, s1, report
    slli t1, t0, 2
    add  t1, s0, t1
    lw   t2, 0(t1)
    bne  t2, t6, find_next
    mv   s3, t0                  # found at index s3
    j    report
find_next:
    addi t0, t0, 1
    j    find

report:
    lw   a0, 0(s0)               # smallest element
    slli t1, s1, 2
    addi t1, t1, -4
    add  t1, s0, t1
    lw   a1, 0(t1)               # largest element
    mv   a2, s2                  # checksum
    mv   a3, s3                  # index of 42
    j    _halt

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
array:
    .word 57, 12, 91,  3, 42, 78, 25, 64
    .word  8, 99, 31, 47, 16, 73, 55, 20

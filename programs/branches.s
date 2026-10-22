# branches.s -- what the branch predictor can and cannot learn.
#
# Three loops with different behaviour. Compare schemes:
#   python -m rvsim run programs/branches.s --bp bimodal
#   python -m rvsim run programs/branches.s --bp gshare
#   python -m rvsim run programs/branches.s --bp not-taken

    .equ HALT, 0x10000004
    .equ N, 40

    .text
_start:
    li   s0, 0                   # running total

    # ---- 1. a long, boringly predictable loop -------------------------
    # Taken N-1 times then not taken. Any 2-bit counter nails this after
    # the first couple of iterations.
    li   t0, N
easy:
    addi s0, s0, 1
    addi t0, t0, -1
    bnez t0, easy

    # ---- 2. a strictly alternating branch -----------------------------
    # Taken, not taken, taken, not taken... A bimodal counter sits in the
    # middle and gets roughly half of them wrong. gshare has the history
    # bits to see the pattern.
    li   t0, N
    li   t1, 0
alt:
    andi t2, t1, 1
    beqz t2, alt_even
    addi s0, s0, 10
    j    alt_next
alt_even:
    addi s0, s0, 100
alt_next:
    addi t1, t1, 1
    addi t0, t0, -1
    bnez t0, alt

    # ---- 3. a data-dependent branch -----------------------------------
    # The direction comes from a table, so no history-based predictor can
    # do better than the bias in the data.
    la   t3, pattern
    li   t0, 16
    li   t1, 0
data:
    add  t4, t3, t1
    lb   t5, 0(t4)
    beqz t5, data_zero
    addi s0, s0, 1000
    j    data_next
data_zero:
    addi s0, s0, 2000
data_next:
    addi t1, t1, 1
    addi t0, t0, -1
    bnez t0, data

    # ---- 4. a call in a loop, to exercise the return-address stack ----
    li   t0, 8
calls:
    li   a0, 3
    call triple
    add  s1, s1, a0
    addi t0, t0, -1
    bnez t0, calls

    j    _halt

triple:
    li   t2, 3
    mul  a0, a0, t2
    ret

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
pattern:
    .byte 1, 0, 0, 1, 1, 1, 0, 1
    .byte 0, 0, 1, 0, 1, 1, 0, 0

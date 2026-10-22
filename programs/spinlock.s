# spinlock.s -- four cores, one cache line, MESI doing real work.
#
#   python -m rvsim run programs/spinlock.s --cores 4
#
# Every hart runs this same image and branches on mhartid. Each takes a
# ticket-free test-and-test-and-set lock built from lr.w/sc.w, increments
# a shared counter, and releases. The final counter must be exactly
# CORES * ITERATIONS -- which it only can be if the coherence protocol
# actually works.
#
# Watch the coherence log: the lock line ping-pongs between cores in
# Modified state, while the read-only "banner" line settles into Shared
# in every L1 at once.

    .equ HALT,       0x10000004
    .equ ITERATIONS, 20

    .text
_start:
    csrr s0, mhartid             # which core am I?
    la   s1, lock
    la   s2, counter
    la   s3, per_core
    li   s4, ITERATIONS

    # ---- a read-only line every core reads: ends up Shared everywhere --
    la   t0, banner
    lw   s5, 0(t0)
    lw   s6, 4(t0)

main_loop:
    beqz s4, done

acquire:
    # Test-and-test-and-set: spin on a plain load first, so the spinning
    # cores sit quietly in Shared instead of fighting over the line with
    # read-for-ownership traffic. Only when it looks free do we try the
    # atomic.
    lw   t0, 0(s1)
    bnez t0, acquire             # still held: keep spinning locally

    li   t1, 1
    amoswap.w t2, t1, (s1)       # atomically take the lock
    bnez t2, acquire             # someone else got there first

critical:
    # ---- the critical section ------------------------------------------
    # A plain load/store pair. It is only safe because of the lock; the
    # counter line also bounces between cores in Modified state.
    lw   t3, 0(s2)
    addi t3, t3, 1
    sw   t3, 0(s2)

    # keep a private tally too, to prove each core did its share
    slli t4, s0, 2
    add  t4, s3, t4
    lw   t5, 0(t4)
    addi t5, t5, 1
    sw   t5, 0(t4)

release:
    # Releasing is a plain store of zero: the line is already Modified in
    # this core, so this costs nothing until another core snoops it away.
    sw   x0, 0(s1)

    addi s4, s4, -1
    j    main_loop

done:
    lw   a0, 0(s2)               # the shared total
    slli t4, s0, 2
    add  t4, s3, t4
    lw   a1, 0(t4)               # this core's own tally
    mv   a2, s0
    j    _halt

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
    .align 5
lock:
    .word 0
    .space 28                    # keep the lock alone in its own line

    .align 5
counter:
    .word 0
    .space 28                    # and the counter alone in its own line

    .align 5
per_core:
    .word 0, 0, 0, 0

    .align 5
banner:
    .word 0x52495343, 0x2D56

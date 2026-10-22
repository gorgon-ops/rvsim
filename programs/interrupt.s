# interrupt.s -- traps, timer interrupts and a WFI park.
#
#   python -m rvsim run programs/interrupt.s --timer 150
#
# The main loop counts. A timer interrupt fires periodically, the handler
# bumps its own counter and rearms the timer, and mret resumes the loop
# exactly where it was -- the counter total proves nothing was lost or
# executed twice.

    .equ HALT,      0x10000004
    .equ MTIMECMP,  0x02004000
    .equ MTIME,     0x0200BFF8
    .equ MIE_MTIE,  0x080          # bit 7:  machine timer interrupt enable
    .equ MSTATUS_MIE, 0x008        # bit 3:  global interrupt enable
    .equ PERIOD,    150

    .text
_start:
    # ---- install the trap vector --------------------------------------
    la   t0, trap_handler
    csrw mtvec, t0

    # ---- arm the timer -------------------------------------------------
    # mtimecmp is 64 bits but this is a 32-bit machine, so it takes two
    # stores. Doing them in the wrong order can make the timer fire
    # spuriously against a half-updated value, so use the sequence the
    # privileged spec recommends: park the low half at all-ones, write the
    # high half, then write the real low half last.
    li   t0, MTIME
    lw   t1, 0(t0)               # read the current time (low 32 bits)
    li   t2, PERIOD
    add  t1, t1, t2
    li   t0, MTIMECMP
    li   t2, -1
    sw   t2, 0(t0)               # 1. low = 0xFFFFFFFF, cannot fire
    sw   x0, 4(t0)               # 2. high = 0
    sw   t1, 0(t0)               # 3. low = the real deadline

    # ---- enable timer interrupts, then interrupts globally ------------
    li   t0, MIE_MTIE
    csrw mie, t0
    li   t0, MSTATUS_MIE
    csrs mstatus, t0

    # ---- a synchronous exception first, to show the difference --------
    # ecall traps immediately and precisely: mepc points at the ecall,
    # so the handler has to step over it before returning.
    li   s0, 0
    ecall
    addi s0, s0, 1               # the handler returns here

    # ---- the interruptible main loop ----------------------------------
    la   s2, counter
    li   t3, 0
    li   t4, 600
work:
    lw   t5, 0(s2)
    addi t5, t5, 1
    sw   t5, 0(s2)               # an interrupt must not repeat this store
    addi t3, t3, 1
    blt  t3, t4, work

    lw   a0, 0(s2)               # should be exactly 600
    mv   a1, s1                  # number of timer interrupts serviced
    mv   a2, s0                  # 1 if the ecall handler returned correctly

    # ---- park the core until the next interrupt ------------------------
    wfi
    j    _halt

# ----------------------------------------------------------------------
# One vector for everything; mcause says which.
# ----------------------------------------------------------------------
# This handler touches only s4-s7, which the main loop never uses, so it
# needs no save/restore prologue. A real kernel would spill to a stack
# found through mscratch.
trap_handler:
    csrr s4, mcause
    bltz s4, interrupt           # the top bit marks an interrupt

exception:
    # Synchronous: mepc points AT the faulting instruction, so skip it.
    csrr s5, mepc
    addi s5, s5, 4
    csrw mepc, s5
    li   s3, 0xEC                # a marker: the exception path ran
    mret

interrupt:
    # Asynchronous: mepc already points at the instruction to resume.
    addi s1, s1, 1               # count the interrupts we serviced
    li   s5, MTIME               # rearm: mtimecmp = mtime + PERIOD
    lw   s6, 0(s5)
    li   s7, PERIOD
    add  s6, s6, s7
    li   s5, MTIMECMP
    li   s7, -1
    sw   s7, 0(s5)               # disarm while we update
    sw   x0, 4(s5)
    sw   s6, 0(s5)               # rearmed; this also clears mip.MTIP
    mret

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
counter:
    .word 0

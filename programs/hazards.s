# hazards.s -- every pipeline hazard the core knows how to handle.
#
# Run with --no-forwarding to watch the same program take far longer:
#   python -m rvsim run programs/hazards.s
#   python -m rvsim run programs/hazards.s --no-forwarding

    .equ HALT, 0x10000004

    .text
_start:
    # ---- back-to-back RAW chain --------------------------------------
    # Every instruction needs the previous result. With forwarding this
    # runs at one instruction per cycle; without it, each pair costs two
    # extra bubbles.
    li   t0, 1
    add  t1, t0, t0          # needs t0 from the instruction before
    add  t2, t1, t1          # needs t1
    add  t3, t2, t2          # needs t2
    add  t4, t3, t3          # t4 = 16

    # ---- load-use hazard ---------------------------------------------
    # Forwarding cannot fix this one: the value does not exist until the
    # end of MEM, so the consumer must wait one cycle no matter what.
    la   s0, table
    lw   a0, 0(s0)
    addi a1, a0, 1           # <- one guaranteed bubble here
    lw   a2, 4(s0)
    add  a3, a2, a1          # <- and here

    # ---- multi-cycle execute -----------------------------------------
    # mul occupies EX for several cycles; div for many more. Everything
    # behind it waits in ID.
    li   t5, 1000
    li   t6, 7
    mul  s1, t5, t6          # 7000
    div  s2, s1, t6          # 1000
    rem  s3, s1, t6          # 0

    # ---- store-to-load through the cache ------------------------------
    sw   s1, 8(s0)
    lw   s4, 8(s0)           # same line, L1 hit
    add  s5, s4, s2          # 8000

    j    _halt

    .text
_halt:
    li   t6, HALT
    sw   x0, 0(t6)
    j    _halt

    .data
table:
    .word 10, 20, 0, 0

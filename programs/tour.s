# tour.s -- a program written to be watched rather than to compute anything.
#
# Every line does one visible, explainable thing, and they are ordered so
# that each of the ideas worth understanding shows up once:
#
#   1. plain arithmetic          - the basic five stages
#   2. back-to-back dependency   - forwarding
#   3. store then load           - the data cache
#   4. load then use             - the one bubble forwarding cannot remove
#   5. a short loop              - branch prediction guessing wrong, then right
#   6. a multiply                - an instruction that occupies EX for a while

    .equ HALT, 0x10000004

    .text
_start:
    # ---- warm up ------------------------------------------------------
    # The very first time the CPU touches anything, it has to go all the
    # way out to main memory, which takes tens of cycles. Getting that out
    # of the way here means the interesting part below runs at full speed.
    la   s0, scratch
    lw   t1, 0(s0)
    nop

    # ---- the tour proper ----------------------------------------------
    li   a0, 6                # put the number 6 into register a0
    li   a1, 7                # put the number 7 into register a1

    add  a2, a0, a1           # 13. a1 was written one cycle ago, so its value
                              #     is forwarded straight into the adder
    sub  a3, a2, a0           # 7.  same again, this time from a2

    sw   a2, 0(s0)            # write 13 into memory
    lw   a4, 0(s0)            # read it back out again
    addi a5, a4, 1            # 14. this one has to wait: the loaded value
                              #     does not exist until the load reaches MEM

    li   t0, 3                # loop three times
    li   a6, 0
sum:
    add  a6, a6, a4           # running total: 13, 26, 39
    addi t0, t0, -1
    bnez t0, sum              # the CPU guesses; the first guess is wrong

    mul  a7, a2, a3           # 91. multiply holds the execute stage for
                              #     several cycles while everything queues up

    j    done

    .text
done:
    li   t6, HALT
    sw   x0, 0(t6)
    j    done

    .data
scratch:
    .word 0

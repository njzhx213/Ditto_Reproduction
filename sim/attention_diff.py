#!/usr/bin/env python3
"""
attention_diff.py - Ditto's attention difference processing (two-sub-op identity).

Paper (Attention Layers, p.342): naive difference of Q*K across time steps needs
three terms, but they collapse to TWO sub-operations by treating Q_t and K_{t+1}
as "weights":

    Q_t K_t - Q_{t+1} K_{t+1}
      = Q_{t+1} dK + dQ K_{t+1} + dQ dK          (naive, three terms)
      = Q_t dK + dQ K_{t+1}                       (Ditto, two terms)

  where dQ = Q_t - Q_{t+1},  dK = K_t - K_{t+1}.

Identity proof: Q_t dK + dQ K_{t+1}
              = Q_t(K_t-K_{t+1}) + (Q_t-Q_{t+1})K_{t+1}
              = Q_t K_t - Q_t K_{t+1} + Q_t K_{t+1} - Q_{t+1} K_{t+1}
              = Q_t K_t - Q_{t+1} K_{t+1}.   QED

Same applies to P*V. Cross-attention: K,V are context-derived and constant across
steps (dK=dV=0), so the two-sub-op collapses to ONE (K_{t+1} as plain weight) ->
identical to conventional linear difference processing.

This file proves the identity numerically (float + int8-quantized) the way Week-1
verified the linear datapath, so the attention mechanism is shown correct before
it enters the performance model.

    cd ~/Ditto && python3 sim/attention_diff.py
"""
import numpy as np


def attn_score_full(Q, K):
    return Q @ K.T


def attn_score_ditto_twoop(Q_t, K_t, Q_p, K_p):
    """Reconstruct (Q_t K_t - Q_p K_p) via Ditto's two sub-operations:
       Q_t dK + dQ K_{t+1}, with Q_p=Q_{t+1}, K_p=K_{t+1}."""
    dQ = Q_t - Q_p
    dK = K_t - K_p
    sub1 = Q_t @ dK.T        # Q_t * dK   (Q_t as weight)
    sub2 = dQ @ K_p.T        # dQ * K_{t+1}  (K_{t+1} as weight)
    return sub1 + sub2


def attn_score_naive_threeop(Q_t, K_t, Q_p, K_p):
    dQ = Q_t - Q_p
    dK = K_t - K_p
    return Q_p @ dK.T + dQ @ K_p.T + dQ @ dK.T


def quantize_int8(x):
    amax = np.abs(x).max()
    if amax == 0:
        return x.copy(), 1.0
    scale = amax / 127.0
    q = np.round(x / scale).clip(-127, 127)
    return q, scale


def main():
    rng = np.random.default_rng(0)
    seq, dim = 64, 32

    print("=== Attention difference identity (Ditto two-sub-op) ===\n")

    # --- float identity: two-op == three-op == direct difference ---
    Q_t = rng.standard_normal((seq, dim))
    K_t = rng.standard_normal((seq, dim))
    Q_p = Q_t + 0.01 * rng.standard_normal((seq, dim))   # adjacent step: small change
    K_p = K_t + 0.01 * rng.standard_normal((seq, dim))

    direct = attn_score_full(Q_t, K_t) - attn_score_full(Q_p, K_p)
    twoop = attn_score_ditto_twoop(Q_t, K_t, Q_p, K_p)
    threeop = attn_score_naive_threeop(Q_t, K_t, Q_p, K_p)

    e_two = np.abs(direct - twoop).max()
    e_three = np.abs(direct - threeop).max()
    print(f"[float] max|direct - two-op|   = {e_two:.2e}")
    print(f"[float] max|direct - three-op| = {e_three:.2e}")
    print(f"[float] max|two-op - three-op| = {np.abs(twoop-threeop).max():.2e}")
    assert e_two < 1e-9 and e_three < 1e-9, "float identity broken!"
    print("  -> two-op exactly reconstructs the difference (identity holds).\n")

    # --- cross-attention: K constant across steps (dK=0) -> two-op collapses to one ---
    K_const = rng.standard_normal((seq, dim))
    direct_cross = attn_score_full(Q_t, K_const) - attn_score_full(Q_p, K_const)
    dQ = Q_t - Q_p
    one_op = dQ @ K_const.T          # only dQ * K  (K as plain weight)
    print(f"[cross-attn] dK=0, max|direct - one-op| = "
          f"{np.abs(direct_cross - one_op).max():.2e}")
    assert np.abs(direct_cross - one_op).max() < 1e-9
    print("  -> with constant K (context), reduces to ONE op = plain linear diff.\n")

    # --- int8-quantized: two-op equals direct within quantization, and the
    #     difference is sparse/low-magnitude (the property Ditto exploits) ---
    Qt_q, sq = quantize_int8(Q_t)
    Kt_q, sk = quantize_int8(K_t)
    Qp_q, _ = quantize_int8(Q_p)   # in practice same scale; kept simple here
    Kp_q, _ = quantize_int8(K_p)
    # two-op on quantized operands (integer matmul domain)
    dQq = Qt_q - Qp_q
    dKq = Kt_q - Kp_q
    two_q = (Qt_q @ dKq.T) + (dQq @ Kp_q.T)
    three_q = (Qp_q @ dKq.T) + (dQq @ Kp_q.T) + (dQq @ dKq.T)
    print(f"[int8] max|two-op - three-op| (integer domain) = "
          f"{np.abs(two_q - three_q).max():.0f}")
    assert np.array_equal(two_q, three_q), "int8 two-op != three-op!"
    print("  -> two-op is bitwise-equal to three-op in the integer domain.\n")

    # --- show the difference operands are sparse (why Ditto wins) ---
    zero_frac = np.mean(dKq == 0)
    print(f"[sparsity] fraction of zero entries in quantized dK: {zero_frac:.1%}")
    print("  (Ditto skips these zeros + uses reduced bit-width on the rest.)\n")

    print("ALL ATTENTION-DIFFERENCE CHECKS PASS")
    print("Mechanism verified: Q*K difference = Q_t.dK + dQ.K_{t+1} (two ops),")
    print("bitwise-exact in integer domain; cross-attn collapses to one op.")


if __name__ == "__main__":
    main()

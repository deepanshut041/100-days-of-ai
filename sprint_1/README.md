Gotcha — here’s a **full day-by-day Sprint 1 plan** *with* coding + math + eval/logging + deliverables, and the exact links you’ll need. It sticks **only** to Sprint 1.

---

# Sprint 1 — Transformer Systems & Tooling (Days 1–10)

**Daily time budget:** `90R/120C/30L` (R=read/watch, C=code/experiments, L=log/eval).
Math-heavy day = `120M/90C/30L`.

---

## Day 1 — Transformer refresher + minimal attention

**Read (90R)**

* *Attention Is All You Need* (skim model, attention, masking, positional encodings). ([arXiv][1])

**Code (120C)**

* Create repo skeleton: `modules/attention.py`, `tests/test_attention.py`.
* Implement **scaled dot-product attention (forward only)** with mask support.
* Unit tests: shapes, masks (causal + padding), NaN/Inf checks.

**Math/Notes (in-line while coding)**

* Derive softmax + cross-entropy gradients in your notebook; note where masking enters the logits.

**Eval/Logging (30L)**

* Log open questions + any numerical stability issues (log-sum-exp trick).

**Deliverables**

* `attention.py` forward + `test_attention.py` green.
* A short note (md) on attention shapes and masking.

---

## Day 2 — Speedups 101 with `torch.compile`

**Read (90R)**

* PyTorch **`torch.compile`** API (what it does, knobs/caveats). ([docs.pytorch.org][2])

**Code (120C)**

* Add a tiny training loop (`train.py`) and wrap model/step with `torch.compile`.
* Add a CLI flag `--compile true|false`.

**Eval/Logging (30L)**

* Benchmark tokens/s and max memory compiled vs baseline (same seeds, 200–500 steps).
* Record versions (PyTorch/CUDA) in logs.

**Deliverables**

* `train.py` (baseline+compiled), benchmark CSV/MD with tokens/s & memory.
* One-paragraph summary: when `torch.compile` helped/hurt. ([docs.pytorch.org][2])

---

## Day 3 — SDPA vs FlashAttention-2 (inference/train microbench)

**Read (90R)**

* PyTorch **SDPA** docs (how backend dispatch works, enable/disable helpers). ([docs.pytorch.org][3])
* *FlashAttention-2* paper + official repo README (supported dtypes/devices, usage constraints). ([arXiv][4], [GitHub][5])

**Code (120C)**

* Add `bench/flashattn_vs_sdpa.py`:

  * seq\_lens = \[512, 1k, 2k, 4k, 8k]
  * adjust batch to fit memory per length
  * measure **forward (decode)** tokens/s; optional train-step tokens/s
  * BF16/FP16 vs FP32 switch
* Toggle SDPA↔FlashAttn via flags; ensure warmup + N timed iters.

**Eval/Logging (30L)**

* Plot `tokens_s_vs_seqlen.png`; summarize stability notes (BF16/FP16).

**Deliverables**

* Bench script + plot + markdown table (mean±std over 3 runs).
* Note SDPA backend chosen per config. ([docs.pytorch.org][3])

---

## Day 4 — KV cache & serving; mock **Paged-KV** API

**Read (90R)**

* vLLM **PagedAttention** paper (motivation, paging, throughput). ([arXiv][6])
* vLLM GitHub (overview of KV paging/serving path). ([GitHub][7])

**Code (120C)**

* Create `kv/mock_paged_kv.py`:

  * `allocate(batch, heads, head_dim, dtype, max_tokens)`
  * `append(b_idx, h_idx, k_chunk, v_chunk)`
  * `get_view(b_idx, h_idx, start, end)` (contiguous view abstraction)
  * `evict(policy="lru")` stub
* Add a tiny **decode** loop that pretends to read from paged storage (no kernel).

**Math/Notes**

* KV bytes per layer: `2 * B * H * T * Dh * bytes_per_elem`; include BF16/FP16/FP32 examples.

**Eval/Logging (30L)**

* Table estimating KV footprint for two configs you care about (e.g., 7B/13B shapes).

**Deliverables**

* Mock paged-KV adaptor and a **KV memory note** (with example numbers). ([arXiv][6])

---

## Day 5 — Numerics sanity: matrix calculus + grad checks

**Read (120M)**

* MIT **18.S096 Matrix Calculus** (lecture notes/OCW page). ([MIT OpenCourseWare][8])
* GELU paper (context for LN/GELU checks). ([arXiv][9])

**Code (90C)**

* Implement **finite-difference grad checks** for **LayerNorm** & **GELU**:

  * `tests/test_gradcheck_ln_gelu.py` with relative error histogram.

**Eval/Logging (30L)**

* Save histogram `plots/grad_error_hist.png`, note ε choices and thresholds.

**Deliverables**

* Grad-check tests + plot + short “what failed first” notes. ([MIT OpenCourseWare][8], [arXiv][9])

---

## Day 6 — MHA cleanup: LayerNorm & determinism

**Read (90R)**

* **Layer Normalization** paper (LN vs BN, per-sample stats). ([arXiv][10])
* PyTorch **reproducibility** notes (seeds, deterministic algos). ([docs.pytorch.org][11])

**Code (120C)**

* Refactor MHA/LN modules: expose `eps`, ensure dtype safety.
* Add `set_determinism.py`: seeds for Python/NumPy/Torch (CPU/GPU), cuDNN knobs.
* Run **3 reproducible training runs** (small config) → capture std-dev of val loss/tokens/s.

**Eval/Logging (30L)**

* Write a determinism checklist you can paste into future projects.

**Deliverables**

* Cleaned modules + **deterministic training script** + table of run-to-run std-dev. ([docs.pytorch.org][11])

---

## Day 7 — Tiny GPT loop (1M tokens baseline)

**Read (90R)**

* PyTorch **Training with PyTorch** tutorial (loop shape: train/eval/save). ([docs.pytorch.org][12])

**Code (120C)**

* Build `models/tiny_gpt.py`: single block config OK.
* `data/` simple tiny corpus → tokenization (char/byte/whitespace).
* Train on **\~1M tokens** with LR warmup + cosine decay (or simple step).
* Save `ckpt.pt` and log **train/val loss** every N steps.

**Eval/Logging (30L)**

* Plot `plots/loss_curve.png` (train vs val).
* Note any overfit/underfit signs and batch/seq impacts.

**Deliverables**

* Baseline checkpoint + loss plot + config used.

---

## Day 8 — Fused ops pass (understand, try, compare)

**Read (90R)**

* **Megatron-Core fusions** overview (what’s typically fused). ([NVIDIA Docs][13])
* NVIDIA **Transformer Engine** user guide (what it accelerates; FP8 context only for awareness). ([NVIDIA Docs][14])

**Code (120C)**

* If your stack supports it, try **fused LayerNorm / bias-GELU** (via available libs or JIT).
* Add `bench/fused_ops.py`: compare fused vs unfused step time over 200 iters.

**Eval/Logging (30L)**

* Table of wall-clock per step; note portability/debug trade-offs.

**Deliverables**

* Benchmark table + short “when to fuse” notes. ([NVIDIA Docs][13])

---

## Day 9 — Positional encodings: RoPE & ALiBi ablation

**Read (90R)**

* **RoPE (RoFormer)** paper. ([arXiv][15])
* **ALiBi** paper (length extrapolation). ([arXiv][16])

**Code (120C)**

* Add `modules/positional.py` with **RoPE** and **ALiBi** toggles.
* Copy-task or tiny next-char task @ short context to sanity-check behavior.
* Configs: `--pos rope|alibi|none`.

**Eval/Logging (30L)**

* Record copy-task accuracy or small val-loss diffs at L={512, 1k}.
* Note “when I’d pick RoPE vs ALiBi” for your later projects.

**Deliverables**

* Positional module + ablation markdown snippet with your observations. ([arXiv][15])

---

## Day 10 — Wrap & doc; make it reproducible

**Read (90R)**

* **pytest** “Get started” (tests discovery, fixtures). ([docs.pytest.org][17])

**Code (120C)**

* Refactor, add docstrings, ensure **`pytest -q`** is green.
* Collect plots (tokens/s vs seq len, loss curve, grad-error hist).
* Make a **single `config.yaml`** and seed-printing banner.

**Eval/Logging (30L)**

* Final README section: environment, commands to reproduce every figure.

**Deliverables**

* Sprint 1 README (this plan + your actual numbers), all plots, green tests.

---

## Quick Link Hub (bookmarks)

* Transformer paper (2017). ([arXiv][1])
* `torch.compile` API. ([docs.pytorch.org][2])
* PyTorch **SDPA** docs. ([docs.pytorch.org][3])
* FlashAttention-2 paper & repo. ([arXiv][4], [GitHub][5])
* vLLM PagedAttention paper & GitHub. ([arXiv][6], [GitHub][7])
* MIT 18.S096 Matrix Calculus (OCW + notes). ([MIT OpenCourseWare][8])
* GELU paper. ([arXiv][9])
* LayerNorm paper; PyTorch reproducibility. ([arXiv][10], [docs.pytorch.org][11])
* Megatron-Core fusions; NVIDIA Transformer Engine guide. ([NVIDIA Docs][13])
* RoPE (RoFormer) & ALiBi. ([arXiv][15])
* pytest quick start. ([docs.pytest.org][17])

---
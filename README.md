\# 100 Days of AI 

\*\*Legend:\*\* Time = \*\*90R/120C/30L\*\* → 90 min read/watch, 120 min code/experiments, 30 min log/eval. On math‑heavy days I flip to \*\*120M/90C/30L\*\*.



---



\## Sprint 1 (Days 1–10) — Transformer Systems \& Tooling Refresher



| Day | Focus                 | Paper/Video                           | Coding/Experiment                            | Math/Notes                           | Eval/Logging                 | Deliverable                      | Time         |

| --: | --------------------- | ------------------------------------- | -------------------------------------------- | ------------------------------------ | ---------------------------- | -------------------------------- | ------------ |

|   1 | Transformer refresher | Transformer paper (2017) skim + notes | Minimal attention forward; unit tests        | Derive softmax + cross‑entropy grads | Note open Qs                 | Minimal attention module + tests | 90R/120C/30L |

|   2 | Speedups 101          | PyTorch 2.x `torch.compile` overview  | Add `torch.compile` to tiny GPT train loop   | N/A                                  | Baseline tokens/s \& memory   | Baseline vs compiled report      | 90R/120C/30L |

|   3 | FlashAttention intro  | FlashAttention summary read           | Swap SDPA↔FlashAttn; microbench              | Stability notes for fp16/bf16        | Tokens/s vs seq len (512→8k) | FlashAttn toggle + bench chart   | 90R/120C/30L |

|   4 | KV cache \& serving    | KV cache/vLLM overview                | Mock paged‑KV adaptor (API only)             | KV footprint estimate                | Decode tokens/s vs batch     | Mock paged‑KV stub + notes       | 90R/120C/30L |

|   5 | Numerics sanity       | Matrix calculus quick recap           | Finite‑diff grad checks (LN, GELU)           | \*\*120M/90C/30L\*\*                     | Gradient error histograms    | Grad‑check notebook              | 120M/90C/30L |

|   6 | MHA cleanup           | Skim LayerNorm \& init best practices  | Clean MHA/LN; set seed; determinism          | Scaling rules note                   | Repro runs ×3, std dev       | Deterministic training script    | 90R/120C/30L |

|   7 | Tiny GPT loop         | Skim training loop patterns           | Build data loader (tiny corpus) + 1M tok run | LR warmup note                       | Train/val loss curve         | Tiny GPT baseline ckpt           | 90R/120C/30L |

|   8 | Fused ops pass        | Read fused kernels overview           | Try fused GELU/LN or fused MLP               | Throughput calc                      | Compare wall clock           | Fused vs unfused table           | 90R/120C/30L |

|   9 | Positional encodings  | RoPE/ALiBi overview                   | Add RoPE; toggle ALiBi                       | Derive RoPE rotation                 | Copy task accuracy           | RoPE/ALiBi ablation config       | 90R/120C/30L |

|  10 | Sprint wrap           | Re‑read notes                         | Refactor + docstrings                        | Math cheat‑sheet                     | Collect plots into README    | Sprint 1 report + cleaned repo   | 90R/120C/30L |



---



\## Sprint 2 (Days 11–20) — Math for DL Research (Targeted)



| Day | Focus               | Paper/Video                | Coding/Experiment                     | Math/Notes               | Eval/Logging                | Deliverable          | Time               |              |

| --: | ------------------- | -------------------------- | ------------------------------------- | ------------------------ | --------------------------- | -------------------- | ------------------ | ------------ |

|  11 | Jacobians           | Matrix calculus mini‑note  | Autograd vs manual grads (softmax/LN) | \*\*120M/90C/30L\*\*         | Unit tests pass/fail        | Jacobian notebook    | 120M/90C/30L       |              |

|  12 | Optimizers          | SGD/Adam behavior overview | LR schedulers (cosine vs constant)    | Step size intuition      | Loss vs steps               | LR schedule plot     | 90R/120C/30L       |              |

|  13 | Init \& norm         | Init/scale notes           | Try Xavier/Kaiming/μParam             | Variance propagation     | Activation stats hist       | Init ablation grid   | 90R/120C/30L       |              |

|  14 | Regularization      | Wd/clip basics             | Add weight decay + grad clip          | Clip rationale           | Grad‑norm traces            | Stability comparison | Reg config + plots | 90R/120C/30L |

|  15 | Convex refresher    | Logistic regression recap  | From‑scratch binary classifier        | Loss convexity           | Train vs closed‑form sanity | Clean example repo   | 90R/120C/30L       |              |

|  16 | Curvature           | Hessian/vector products    | HVP via autograd; eigval estimate     | \*\*120M/90C/30L\*\*         | Spectrum snapshot           | HVP utility          | 120M/90C/30L       |              |

|  17 | Numerical stability | Log‑sum‑exp, log‑softmax   | Swap to stable log‑softmax everywhere | Overflow/underflow notes | NaN guardrails pass         | Stable kernels PR    | 90R/120C/30L       |              |

|  18 | CE gradient         | CE derivation              | Manual ∂L/∂x check vs autograd        | \*\*120M/90C/30L\*\*         | ε‑check table               | Derivation page      | 120M/90C/30L       |              |

|  19 | Stability study     | Pull all knobs together    | Run 4 configs side‑by‑side            | Notes on curvature vs LR | Best config selection       | Stability memo       | 90R/120C/30L       |              |

|  20 | Sprint wrap         | Summarize math takeaways   | Clean helpers \& tests                 | Glossary                 | Collate figures             | Sprint 2 report      | 90R/120C/30L       |              |



---



\## Sprint 3 (Days 21–30) — Pretraining Pipeline \& Scaling



| Day | Focus          | Paper/Video                        | Coding/Experiment                       | Math/Notes            | Eval/Logging           | Deliverable               | Time         |

| --: | -------------- | ---------------------------------- | --------------------------------------- | --------------------- | ---------------------- | ------------------------- | ------------ |

|  21 | Tokenizer      | Subword/tokenizer overview         | Train SentencePiece (16k/32k)           | Zipf law note         | OOV/len stats          | Tokenizer configs + vocab | 90R/120C/30L |

|  22 | Data loader    | Streaming/sharding overview        | Build streaming dataloader (memmap/web) | I/O throughput calc   | Loader perf metrics    | Streaming loader module   | 90R/120C/30L |

|  23 | Scaling laws   | Read compute/data/params tradeoffs | Draft budget for 20–50M model           | Loss vs compute model | Target tokens budget   | Scaling plan doc          | 90R/120C/30L |

|  24 | Config design  | Model shape choices                | Set n\\\_layer/n\\\_head/d\\\_model/seq       | Param count math      | Param/token calc       | Config YAMLs              | 90R/120C/30L |

|  25 | Start pretrain | N/A                                | Kick off 20–50M pretrain (ckpting ON)   | LR schedule note      | Live logs (wandb/text) | First ckpt + logs         | 90R/120C/30L |

|  26 | Validation     | Perplexity basics                  | Add val loop + early stop               | CE↔PPL relation       | PPL curve              | Val script                | 90R/120C/30L |

|  27 | Vocab ablation | N/A                                | 16k vs 32k tokenizer runs               | Token efficiency note | PPL @ fixed steps      | Ablation table            | 90R/120C/30L |

|  28 | Throughput     | Pipeline stalls overview           | Micro‑batch/accumulation tuning         | Amdahl’s law note     | Tokens/s deltas        | Throughput report         | 90R/120C/30L |

|  29 | LR schedules   | Cosine vs 1‑cycle                  | Swap schedules                          | Intuition on warmup   | Loss slope compare     | Best LR schedule          | 90R/120C/30L |

|  30 | Sprint wrap    | Re‑read notes                      | Refactor + save best config             | Summary math          | Curves + decision      | Sprint 3 report           | 90R/120C/30L |



---



\## Sprint 4 (Days 31–40) — Efficient Attention \& Long Context



| Day | Focus          | Paper/Video            | Coding/Experiment                  | Math/Notes           | Eval/Logging        | Deliverable           | Time         |

| --: | -------------- | ---------------------- | ---------------------------------- | -------------------- | ------------------- | --------------------- | ------------ |

|  31 | MQA            | MQA overview           | Implement MQA path                 | Param count diff     | Train speed compare | MQA switch            | 90R/120C/30L |

|  32 | GQA            | GQA overview           | Implement GQA                      | Memory model note    | Peak mem vs seq     | GQA toggle            | 90R/120C/30L |

|  33 | FlashAttn long | Long‑seq tricks        | Long‑seq bench (512→8k)            | Complexity recap     | Tokens/s chart      | Long‑seq bench script | 90R/120C/30L |

|  34 | KV decode      | Serving patterns       | Decode‑only loop + KV reuse        | Cache size math      | Decode tokens/s     | Decode bench          | 90R/120C/30L |

|  35 | Paging sim     | Paging intuition       | Simulate paged‑KV fragmentation    | Fragmentation note   | Latency variance    | Paging sim notebook   | 90R/120C/30L |

|  36 | Positional     | RoPE/ALiBi recap       | Add ALiBi path                     | Rotations math       | Copy‑task accuracy  | RoPE vs ALiBi plot    | 90R/120C/30L |

|  37 | Needle test    | Retrieval eval pattern | Build Needle‑in‑Haystack generator | Retrieval prob model | Accuracy vs length  | Needle eval harness   | 90R/120C/30L |

|  38 | Retention      | N/A                    | Run retention tests 1k→32k         | Decay curve note     | Hit‑rate plots      | Retention curves      | 90R/120C/30L |

|  39 | Choice         | N/A                    | Pick best variant(s) for later     | Trade‑off table      | Rationale log       | Choice memo           | 90R/120C/30L |

|  40 | Sprint wrap    | Review                 | Merge + doc                        | Summary math         | Collate charts      | Sprint 4 report       | 90R/120C/30L |



---



\## Sprint 5 (Days 41–50) — Post‑Training: SFT → DPO → PPO‑lite



| Day | Focus        | Paper/Video         | Coding/Experiment                                   | Math/Notes         | Eval/Logging            | Deliverable            | Time         |

| --: | ------------ | ------------------- | --------------------------------------------------- | ------------------ | ----------------------- | ---------------------- | ------------ |

|  41 | SFT data     | SFT overview        | Prepare small instruction dataset (you have rights) | N/A                | Data stats \& splits     | Clean dataset + card   | 90R/120C/30L |

|  42 | SFT loop     | SFT recipes         | Train SFT; overfit 200 examples to sanity‑check     | Loss decomposition | Train/val curves        | SFT checkpoint         | 90R/120C/30L |

|  43 | DPO          | DPO concept         | Implement pairwise loss; prompt→(good,bad)          | Preference math    | Reward proxy notes      | DPO trainer            | 90R/120C/30L |

|  44 | SFT vs DPO   | N/A                 | Compare outputs on held‑out prompts                 | KL/divergence note | Human quick ratings     | Comparison table       | 90R/120C/30L |

|  45 | PPO‑lite     | PPO intuition       | Simple PPO loop (small rollout)                     | Advantage estimate | Reward/entropy curves   | PPO‑lite script        | 90R/120C/30L |

|  46 | Safety style | Refusal styles      | Add simple policy rules in decoding                 | Calibration idea   | Safety probe set        | Safety notes + prompts | 90R/120C/30L |

|  47 | Decoding     | Sampling strategies | Implement temp/top‑p/top‑k                          | Prob mass note     | Diversity/quality sweep | Decoding toolkit       | 90R/120C/30L |

|  48 | Eval set     | Evaluation prompts  | Build small eval battery                            | Metric choices     | Exact‑match/len/latency | Eval harness v1        | 90R/120C/30L |

|  49 | KL ablation  | Reg strength study  | Vary KL on DPO/PPO                                  | Bias‑variance note | Win‑rate vs KL          | KL→quality plot        | 90R/120C/30L |

|  50 | Sprint wrap  | Review              | Clean code \& scripts                                | Summary            | Collate results         | Sprint 5 report        | 90R/120C/30L |



---



\## Sprint 6 (Days 51–60) — Evaluation \& Alignment



| Day | Focus         | Paper/Video              | Coding/Experiment                      | Math/Notes           | Eval/Logging        | Deliverable           | Time         |

| --: | ------------- | ------------------------ | -------------------------------------- | -------------------- | ------------------- | --------------------- | ------------ |

|  51 | Harness setup | Eval frameworks overview | Wire up task runner                    | Metric definitions   | Baseline scores     | Harness v1            | 90R/120C/30L |

|  52 | Tasks         | Task taxonomy            | Add tasks (QA, summarization small)    | Metric pitfalls      | Per‑task logs       | Tasks config          | 90R/120C/30L |

|  53 | Calibration   | Calibration overview     | Temperature scaling for classification | Reliability math     | Reliability diagram | Calibrated classifier | 90R/120C/30L |

|  54 | Safety probes | Bias/toxicity probes     | Add simple probe sets                  | Measurement caveats  | Probe scores        | Probe scripts         | 90R/120C/30L |

|  55 | Repro runs    | N/A                      | One‑click eval script                  | Variance tracking    | CI‑like check       | Eval CLI              | 90R/120C/30L |

|  56 | Unit tests    | Testing in ML            | Tests for tokenization, decoding, eval | Test coverage note   | Pass/fail table     | Tests passing         | 90R/120C/30L |

|  57 | Scoreboard    | N/A                      | Aggregate past runs into table         | Significance note    | Best/worst cases    | Scoreboard markdown   | 90R/120C/30L |

|  58 | Visualization | Plotting basics          | Matplotlib plots for results           | Confidence intervals | Summary figs        | Plots assets folder   | 90R/120C/30L |

|  59 | Eval card     | Model cards              | Draft eval card template               | Reporting checklists | Versioned results   | Eval card v1          | 90R/120C/30L |

|  60 | Sprint wrap   | Review                   | Refactor + docs                        | Summary              | Curate artifacts    | Sprint 6 report       | 90R/120C/30L |



---



\## Sprint 7 (Days 61–70) — Alternative Architectures (SSMs / Mamba‑style)



| Day | Focus          | Paper/Video           | Coding/Experiment               | Math/Notes             | Eval/Logging      | Deliverable           | Time         |

| --: | -------------- | --------------------- | ------------------------------- | ---------------------- | ----------------- | --------------------- | ------------ |

|  61 | SSM intro      | SSM/Mamba overview    | Draft minimal SSM block         | State update math      | Sanity checks     | SSM skeleton          | 90R/120C/30L |

|  62 | Selective scan | Implementation detail | Implement selective scan        | Stability note         | Train step time   | Working SSM block     | 90R/120C/30L |

|  63 | GPT parity     | N/A                   | Char‑LM: SSM vs GPT same tokens | Apples‑to‑apples setup | PPL vs time       | Parity benchmark      | 90R/120C/30L |

|  64 | Long seq       | Long‑range limits     | Scale seq 2k→32k                | Complexity compare     | Tokens/s vs seq   | Long‑seq compare plot | 90R/120C/30L |

|  65 | Memory         | Memory models         | Measure peak mem vs batch/seq   | Memory math            | Peak tables       | Mem profile notes     | 90R/120C/30L |

|  66 | Gating         | Design variants       | Add gating/skip                 | Stability insights     | PPL deltas        | Gated SSM variant     | 90R/120C/30L |

|  67 | Alg tasks      | Algorithmic tasks     | Copy/sort toy datasets          | Generalization note    | Acc/len curves    | Task scripts          | 90R/120C/30L |

|  68 | Analysis       | N/A                   | Error taxonomy                  | Inductive bias note    | Fail case catalog | Analysis doc          | 90R/120C/30L |

|  69 | Decision       | N/A                   | Keep SSM path or GPT focus      | Trade‑off summary      | Rationale         | Decision memo         | 90R/120C/30L |

|  70 | Sprint wrap    | Review                | Merge + doc                     | Summary                | Collate plots     | Sprint 7 report       | 90R/120C/30L |



---



\## Sprint 8 (Days 71–80) — Diffusion Models



| Day | Focus            | Paper/Video               | Coding/Experiment            | Math/Notes                  | Eval/Logging        | Deliverable            | Time         |

| --: | ---------------- | ------------------------- | ---------------------------- | --------------------------- | ------------------- | ---------------------- | ------------ |

|  71 | DDPM forward     | DDPM overview             | Implement forward noising    | KL intuition                | Recon loss curve    | DDPM core              | 90R/120C/30L |

|  72 | DDPM reverse     | Sampling steps            | Train on MNIST/CIFAR‑small   | Variance schedule           | Sample grid v0      | Samples + script       | 90R/120C/30L |

|  73 | DDIM             | DDIM intro                | Add DDIM sampler             | Deterministic vs stochastic | Steps vs quality    | DDIM sampler           | 90R/120C/30L |

|  74 | CFG              | Classifier‑free guidance  | Implement CFG                | Guidance scale math         | Quality vs guidance | CFG plots              | 90R/120C/30L |

|  75 | Latent diffusion | VAE + U‑Net latent        | Tiny VAE; encode→diffuse     | Rate‑distortion note        | Recon metrics       | VAE + latent pipeline  | 90R/120C/30L |

|  76 | Conditioning     | Conditioning methods      | Label/text conditioning stub | Conditioning math           | Cond vs uncond      | Cond models            | 90R/120C/30L |

|  77 | Steps ablation   | N/A                       | Steps 10→100 compare         | Time/quality tradeoff       | Runtime table       | Ablation report        | 90R/120C/30L |

|  78 | Schedules        | Schedules (linear/cosine) | Implement schedule toggles   | Noise schedule note         | FID‑lite proxy      | Schedule compare       | 90R/120C/30L |

|  79 | Packaging        | N/A                       | Save pipelines/checkpoints   | Repro check                 | Release assets      | Diffusion mini‑release | 90R/120C/30L |

|  80 | Sprint wrap      | Review                    | Doc + cleanup                | Summary                     | Gallery + charts    | Sprint 8 report        | 90R/120C/30L |



---



\## Sprint 9 (Days 81–90) — Systems Scaling (FSDP/ZeRO, LoRA/QLoRA, Serving)



| Day | Focus         | Paper/Video             | Coding/Experiment           | Math/Notes          | Eval/Logging        | Deliverable           | Time         |

| --: | ------------- | ----------------------- | --------------------------- | ------------------- | ------------------- | --------------------- | ------------ |

|  81 | FSDP/ZeRO     | Dist training overview  | Wrap model with FSDP/ZeRO‑2 | Shard math          | Throughput/mem logs | Dist config           | 90R/120C/30L |

|  82 | Precision     | Mixed precision         | bf16/fp16 + grad scaling    | Rounding error note | Stability vs speed  | Precision settings    | 90R/120C/30L |

|  83 | Checkpointing | Activation ckpt/offload | Add ckpt + CPU offload      | Memory calc         | Peak/step time      | Memory‑speed tradeoff | 90R/120C/30L |

|  84 | LoRA          | Adapter methods         | Implement LoRA              | Rank/α intuition    | PPL vs rank         | LoRA trainer          | 90R/120C/30L |

|  85 | QLoRA         | 4‑bit adapters          | Implement QLoRA path        | Quant error note    | Quality vs speed    | QLoRA trainer         | 90R/120C/30L |

|  86 | Compare FT    | Fine‑tune vs adapters   | Run FT vs LoRA vs QLoRA     | Cost model          | Win‑rate table      | Comparison doc        | 90R/120C/30L |

|  87 | Serving       | Serving overview        | Try vLLM‑style serving      | Latency math        | Concurrency scaling | Serving script        | 90R/120C/30L |

|  88 | Red‑team lite | Prompt safety basics    | Simple red‑team prompts     | Guardrails caveats  | Failure tally       | Safety notes          | 90R/120C/30L |

|  89 | Costing       | Compute accounting      | Tokens/energy/\\$ estimates  | Scaling law revisit | Cost table          | Costing sheet         | 90R/120C/30L |

|  90 | Sprint wrap   | Review                  | Cleanup + doc               | Summary             | Curate comparisons  | Sprint 9 report       | 90R/120C/30L |



---



\## Sprint 10 (Days 91–100) — Capstone Mini‑Research



| Day | Focus             | Paper/Video                  | Coding/Experiment                        | Math/Notes             | Eval/Logging            | Deliverable           | Time         |

| --: | ----------------- | ---------------------------- | ---------------------------------------- | ---------------------- | ----------------------- | --------------------- | ------------ |

|  91 | Topic + design    | Pick A/B/C topic \& baselines | Repo scaffold; baseline run              | Hypotheses             | Baseline results        | Plan + baseline       | 90R/120C/30L |

|  92 | Ablation grid     | N/A                          | Implement grid (e.g., ranks, heads, seq) | Power analysis note    | Run schedule            | Grid config           | 90R/120C/30L |

|  93 | Experiments I     | N/A                          | Run batch 1                              | Error bars note        | Results table v1        | Checkpoint set 1      | 90R/120C/30L |

|  94 | Experiments II    | N/A                          | Run batch 2                              | Significance test note | Results table v2        | Checkpoint set 2      | 90R/120C/30L |

|  95 | Write: Methods    | Related work skim            | Write Intro/Methods                      | Clarity checklist      | Draft sections          | Paper draft I         | 90R/120C/30L |

|  96 | Write: Results    | N/A                          | Write Results + figs                     | Stats recap            | Finalize plots          | Paper draft II        | 90R/120C/30L |

|  97 | Write: Disc/Limit | N/A                          | Write Discussion/Limitations             | Threats to validity    | Reviewer Qs             | Paper draft III       | 90R/120C/30L |

|  98 | Repro repo        | N/A                          | Clean repo; scripts; configs             | Repro checklist        | Fresh run passes        | Reproducible repo     | 90R/120C/30L |

|  99 | Polish            | N/A                          | Edit paper; create slides                | Storyline notes        | Dry‑runs log            | Camera‑ready + slides | 90R/120C/30L |

| 100 | Present           | N/A                          | Final run + presentation                 | Takeaways              | Audience feedback notes | Final release \& talk  | 90R/120C/30L |



---


# Sprint 0 — PyTorch Advanced Research Roadmap

This sprint is about refreshing PyTorch fluency and building up the **end-to-end skills needed for LLM research and scaling**.
Each module builds toward a **capstone: training, scaling, exporting, and analyzing a small GPT-style language model**.

---

## 📦 M0 — Quick PyTorch refresh (syntax & workflow)

**Goal:** Be fluent again with tensors, `nn.Module`, training loops, saving/loading.

* [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/index.html?utm_source=chatgpt.com)
* [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html?utm_source=chatgpt.com)

✅ **Outcome:** Write clean training loops with proper checkpointing.

---

## 🗂 M1 — Data pipelines you can trust at scale

**Goal:** Rock-solid `Dataset`/`DataLoader`, plus custom datasets and modern text pipelines.

* [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html?utm_source=chatgpt.com)
* [Writing Custom Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial?utm_source=chatgpt.com)
* (Optional) [TorchText custom dataset + DataPipes](https://docs.pytorch.org/tutorials/_downloads/e80c8c5b8a71514d0905366c448448c0/torchtext_custom_dataset_tutorial.py?utm_source=chatgpt.com)

✅ **Outcome:** Clean, testable pipelines; know when to customize collate functions.

---

## 🔄 M2 — Autograd: from basics to custom gradients

**Goal:** Read/modify backprop logic for research.

* [Autograd Fundamentals](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html?utm_source=chatgpt.com)
* [Custom Autograd Functions](https://docs.pytorch.org/tutorials//beginner/examples_autograd/polynomial_custom_function.html?utm_source=chatgpt.com)
* [Double Backward](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html?utm_source=chatgpt.com)

✅ **Outcome:** Prototype new losses/ops with custom backward logic.

---

## ⚡ M3 — Profiling & performance hygiene

**Goal:** Find bottlenecks, use AMP, adopt best practices.

* [Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html?utm_source=chatgpt.com)
* [TensorBoard Profiler](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?utm_source=chatgpt.com)
* [Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html?utm_source=chatgpt.com)
* [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html?utm_source=chatgpt.com)

✅ **Outcome:** Baseline throughput & memory efficiency.

---

## 🚀 M4 — Compilers & graph tooling

**Goal:** Speed up with PyTorch 2.x compilers.

* [Intro to `torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html?utm_source=chatgpt.com)
* [Compiled Autograd](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html?utm_source=chatgpt.com)
* [Compile-time Caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html?utm_source=chatgpt.com)
* [Compilers Index](https://docs.pytorch.org/tutorials/compilers_index.html?utm_source=chatgpt.com)

✅ **Outcome:** Apply `torch.compile` speedups; know when to export.

---

## 🧩 M5 — Transformer & attention building blocks

**Goal:** Ground in Transformer stack & fast attention.

* [Language Modeling with `nn.Transformer`](https://docs.pytorch.org/tutorials/_downloads/aa3898eb04d468790e00cb42405b1c23/transformer_tutorial.py?utm_source=chatgpt.com)
* [Scaled Dot Product Attention](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html?utm_source=chatgpt.com)
* [Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html?utm_source=chatgpt.com)

✅ **Outcome:** Implement/modify Transformer layers with SDPA.

---

## 🌐 M6 — Distributed training (DDP → FSDP2 → TP → PP)

**Goal:** Train multi-billion-param models reliably.

* [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html?utm_source=chatgpt.com)
* [Writing Distributed Apps](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html?utm_source=chatgpt.com)
* [DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=chatgpt.com)
* [FSDP2 Getting Started](https://docs.pytorch.org/tutorials//intermediate/FSDP_tutorial.html?utm_source=chatgpt.com)
* [FSDP Advanced](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html?utm_source=chatgpt.com)
* [Tensor Parallel](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html?utm_source=chatgpt.com)
* [Pipeline Parallelism](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html?utm_source=chatgpt.com)
* [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html?utm_source=chatgpt.com)
* [Distributed Checkpointing](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html?utm_source=chatgpt.com)

✅ **Outcome:** Mix DDP, FSDP2, TP, PP; resume reliably with DCP.

---

## 📤 M7 — Export, quantization & deployment

**Goal:** Prep models for fast inference/interop.

* [torch.export Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html?utm_source=chatgpt.com)
* [Export Challenges & Solutions](https://docs.pytorch.org/tutorials/recipes/torch_export_challenges_solutions.html?utm_source=chatgpt.com)
* [Post-Training Quantization](https://docs.pytorch.org/tutorials/prototype/pt2e_quant_ptq.html?utm_source=chatgpt.com)
* [Quantization-Aware Training](https://docs.pytorch.org/tutorials/prototype/pt2e_quant_qat.html?utm_source=chatgpt.com)
* [Export to ONNX](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html?utm_source=chatgpt.com)

✅ **Outcome:** Export & quantize models; debug export gaps.

---

## 🔬 M8 — Research utilities: `torch.func`, vmap, FX

**Goal:** Experiment faster with Jacobians, PSG, ensembling.

* [Jacobians/Hessians](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html?utm_source=chatgpt.com)
* [Per-Sample Gradients](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html?utm_source=chatgpt.com)
* [Model Ensembling with vmap](https://docs.pytorch.org/tutorials/intermediate/ensembling.html?utm_source=chatgpt.com)
* [FX Profiling](https://docs.pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html?utm_source=chatgpt.com)

✅ **Outcome:** Prototype PSG, NTK analyses, or FX passes.

---

## 🎯 M9 — Reproducibility & experiment rigor

**Goal:** Determinism when you need it.

* [Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html?utm_source=chatgpt.com)
* [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html?utm_source=chatgpt.com)

✅ **Outcome:** Flip deterministic modes responsibly; explain trade-offs.

---

## 🏆 Capstone Path

1. **Implement & train a small GPT-style LM** (M5 + M1).
2. **Profile & optimize** (M3–M4).
3. **Scale to multi-GPU with FSDP2 + TP** (M6).
4. **Export & quantize for inference** (M7).
5. **Add research instrumentation** (M8).

---

## ⚠️ Notes & Guardrails

* `torch.export` and some `vmap` features are **prototype**; expect changes.
* **TorchScript** is deprecated → use `torch.compile` and `torch.export`.

---

Would you like me to also make this into a **GitHub-style checklist version** (so you can tick off each tutorial as you complete it)?

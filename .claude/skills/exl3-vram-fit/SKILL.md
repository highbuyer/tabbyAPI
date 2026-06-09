---
name: exl3-vram-fit
description: Find the largest exllamav3/tabbyAPI config (cache_size/context, max_batch_size, draft/MTP) that fits in GPU VRAM, WITHOUT trial-and-error server restarts. Use when a model OOMs on load ("Insufficient VRAM in split for model and cache"), when picking cache_size/max_batch_size for a new model or quant, or when enabling MTP/draft speculative decoding and load fails. Runs an offline probe that loads the model exactly the way tabbyAPI does and reports fit via reliable VRAM measurement.
---

# exl3-vram-fit

Stop the restart→OOM→tweak loop. This probes candidate configs offline (the model
loads in ~2–5s for exl3) and reports which fit, so you change `config.yml` once.

## Key facts (do not relearn these)

- **Measure VRAM with `torch.cuda.mem_get_info()` only.** `nvidia-smi` and
  `torch.cuda.memory_allocated()` are both fooled by exl3's `cudaMallocAsync`
  pool / raw allocations and will read stale or zero.
- **The probe LOADS the model (~20GB for a 27B@6bpw).** The GPU must be free.
  If tabbyAPI is serving this model, unload it or stop the server first, else the
  probe OOMs on contention (a false negative).
- **Recurrent / hybrid-attention models (e.g. Qwen3.5/3.6: linear + full attn)
  are the trap.** With draft/MTP, the cache stores recurrent-state rollback
  history ≈ `max_history(=draft_num_tokens) × linear_layers × state × max_batch_size`.
  This — not the KV cache — is usually what OOMs, and it scales with
  **max_batch_size**. So `max_batch_size` is the first lever, not `cache_size`.
  tabbyAPI defaults `max_batch_size` to 4 for recurrent models; **1** is often
  required to fit long context + MTP on 24GB.
- **Load order matters:** caches are created and attached BEFORE weights load, so
  the autosplit loader reserves them during load. The probe replicates this.
- **Single GPU:** `autosplit_no_forward=True` skips the calibration forward whose
  logits peak (chunk × vocab) can OOM an otherwise-fitting model. The probe and
  the patched backend both set this.

## Procedure

1. **Confirm the GPU is free:**
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
   ```
   If a server holds the model, stop it (or unload via the admin API) first.
   Also kill stray probe interpreters from earlier runs (`python.exe -`).

2. **Identify the target** from the user / current `config.yml` + start args:
   - model dir, desired context (`cache_size`, usually = `max_seq_len`)
   - `cache_mode` (e.g. `Q4`), draft mode (`mtp`, a draft dir, or `none`),
     `draft_cache_mode`.

3. **Probe candidates** (one subprocess each — CUDA state is unreliable after an
   OOM, so never reuse a process). Use the project venv python:
   ```bash
   venv/Scripts/python.exe .claude/skills/exl3-vram-fit/probe.py \
     --model-dir models/<MODEL> \
     --cache-size 131072 --cache-mode Q4 \
     --draft mtp --draft-cache-mode Q4 \
     --max-batch-size 1 --chunk-size 512
   ```
   Each run prints a final `RESULT_JSON {...}` line with `fits`, `free_after_gb`,
   and on failure `oom`/`error`/`free_at_fail_gb`.

4. **Search strategy** (cheap — each load is seconds):
   - Start at the user's desired `cache_size` with `max_batch_size=1`.
   - **If it fits:** report headroom (`free_after_gb`). If they want concurrency,
     bump `max_batch_size` (2,4,…) until it stops fitting; report the max.
   - **If it OOMs:** lower `max_batch_size` to 1 first. Still OOM → reduce
     `cache_size` (binary search between a known-fit small value and the target).
     `chunk_size` (e.g. 512) trims the load peak as a secondary lever.
   - Leave ≥ ~0.8–1.0 GB headroom (`free_after_gb`) for runtime activation.

5. **Report the winning config** as concrete `config.yml` keys, e.g.:
   ```yaml
   model:
     cache_size: 131072
     max_batch_size: 1
     chunk_size: 512
   draft_model:
     draft_mtp: true
     draft_cache_mode: Q4
   ```
   Note `config.yml` is gitignored here — these stay local.

## Notes

- For a non-MTP standard draft model, pass `--draft <draft_model_dir>` instead of
  `--draft mtp`.
- `--draft none` probes the base model alone (useful to isolate how much the
  draft/MTP actually costs).
- See the project memory `mtp-exl3-setup` for how MTP is wired into this repo.

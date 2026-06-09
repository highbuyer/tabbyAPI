#!/usr/bin/env python
"""
exl3-vram-fit probe — load ONE exllamav3 config exactly the way tabbyAPI does
and report whether it fits in VRAM, using reliable torch.cuda.mem_get_info().

Run ONE config per process (CUDA state is unreliable after an OOM), then let the
caller sweep. Prints human-readable lines plus a final JSON line for parsing.

Why this mirrors tabbyAPI (backends/exllamav3/model.py) and not a naive load:
  * caches are created (and attached to the model) BEFORE loading weights, so the
    autosplit loader reserves them during load — loading first then allocating
    cache under-reports usage and gives wrong "fits" answers.
  * MTP draft = the model's own "mtp" component from the SAME dir; its cache uses
    max_history = draft_num_tokens, which for recurrent/linear-attention models
    dominates VRAM and scales with max_batch_size (the real OOM driver).
  * single GPU passes autosplit_no_forward=True (skips the calibration forward
    whose logits peak can OOM an otherwise-fitting model).

Measure ONLY with torch.cuda.mem_get_info(); nvidia-smi and
torch.cuda.memory_allocated() are both fooled by exl3's cudaMallocAsync pool.
"""
import argparse, json, re, sys, traceback


def gb_free_used():
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, (total - free) / 1e9


def parse_cache_mode(mode):
    """Return (k_bits, v_bits) for a quant cache, or None for FP16."""
    m = {"Q4": (4, 4), "Q6": (6, 6), "Q8": (8, 8)}.get(mode)
    if m:
        return m
    g = re.match(r"^\s*([2-8])\s*,\s*([2-8])\s*$", mode)
    if g:
        return int(g.group(1)), int(g.group(2))
    if mode.upper() == "FP16":
        return None
    raise ValueError(f"Unrecognized cache mode: {mode!r} (use FP16/Q4/Q6/Q8 or 'k,v')")


def make_cache(model, cache_size, kv, max_batch_size, max_history):
    from exllamav3 import Cache
    kwargs = dict(max_num_tokens=cache_size)
    # tabby passes max_batch_size + max_history when the Cache supports them
    import inspect
    if "max_batch_size" in inspect.signature(Cache.__init__).parameters:
        kwargs["max_batch_size"] = max_batch_size
        kwargs["max_history"] = max_history
    if kv is not None:
        from exllamav3.cache import CacheLayer_quant
        kwargs.update(layer_type=CacheLayer_quant, k_bits=kv[0], v_bits=kv[1])
    return Cache(model, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cache-size", type=int, required=True)
    ap.add_argument("--cache-mode", default="Q4", help="FP16/Q4/Q6/Q8 or 'k,v' (e.g. 4,4)")
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--draft", default="none", help="none | mtp | <draft_model_dir>")
    ap.add_argument("--draft-cache-mode", default="Q4")
    ap.add_argument("--draft-num-tokens", type=int, default=0,
                    help="0 = use draft model's default_draft_size")
    ap.add_argument("--reserve-mb", type=float, default=96.0)
    args = ap.parse_args()

    result = {"fits": False, "cache_size": args.cache_size,
              "max_batch_size": args.max_batch_size, "chunk_size": args.chunk_size,
              "draft": args.draft, "failed_at_module": None, "error": None}
    try:
        import torch
        from exllamav3 import Config, Model

        kv = parse_cache_mode(args.cache_mode)
        dkv = parse_cache_mode(args.draft_cache_mode)
        reserve = [args.reserve_mb / 1024]
        single_gpu = torch.cuda.device_count() == 1

        f0, u0 = gb_free_used()
        print(f"start: free={f0:.2f}G used={u0:.2f}G  (device_count={torch.cuda.device_count()})")

        cfg = Config.from_directory(args.model_dir)
        model = Model.from_config(cfg, component="text")

        # ---- draft model ----
        draft_model = None
        ndt = 0
        if args.draft == "mtp":
            if "mtp" not in cfg.model_classes:
                raise RuntimeError("model has no 'mtp' component (need mtp_num_hidden_layers>0)")
            draft_model = Model.from_config(cfg, component="mtp")
            ndt = args.draft_num_tokens or draft_model.caps.get("default_draft_size", 4)
        elif args.draft != "none":
            dcfg = Config.from_directory(args.draft)
            draft_model = Model.from_config(dcfg)
            ndt = args.draft_num_tokens or draft_model.caps.get("default_draft_size", 4)
        result["draft_num_tokens"] = ndt

        # ---- caches FIRST (tabby order), attached to models ----
        make_cache(model, args.cache_size, kv, args.max_batch_size, ndt)
        if draft_model is not None:
            make_cache(draft_model, args.cache_size, dkv, args.max_batch_size, ndt)

        load_kw = dict(reserve_per_device=reserve, max_chunk_size=args.chunk_size)
        import inspect
        lsig = inspect.signature(model.load_gen).parameters
        if "max_batch_size" in lsig:
            load_kw["max_batch_size"] = args.max_batch_size
        if "autosplit_no_forward" in lsig and single_gpu:
            load_kw["autosplit_no_forward"] = True

        # ---- load draft, then main (tabby order) ----
        if draft_model is not None:
            for _ in draft_model.load_gen(**load_kw):
                pass
            torch.cuda.synchronize()
            f, u = gb_free_used()
            print(f"after draft weights: free={f:.2f}G used={u:.2f}G")

        last = 0
        for v in model.load_gen(**load_kw):
            if isinstance(v, (tuple, list)) and v:
                last = v[0]
        torch.cuda.synchronize()
        f, u = gb_free_used()
        result.update(fits=True, free_after_gb=round(f, 2), used_after_gb=round(u, 2))
        print(f"after main weights: free={f:.2f}G used={u:.2f}G  => FITS")

    except Exception as e:
        is_oom = (e.__class__.__name__ == "OutOfMemoryError"
                  or "out of memory" in str(e).lower()
                  or "Insufficient VRAM" in str(e))
        result["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        result["oom"] = bool(is_oom)
        try:
            f, u = gb_free_used()
            result["free_at_fail_gb"] = round(f, 2)
        except Exception:
            pass
        print(f"FAILED ({'OOM' if is_oom else 'error'}): {result['error']}", file=sys.stderr)
        if not is_oom:
            traceback.print_exc()

    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()

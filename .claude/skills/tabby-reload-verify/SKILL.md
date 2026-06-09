---
name: tabby-reload-verify
description: Restart this tabbyAPI server, wait for the model to finish loading, run a smoke generation, and report tokens/sec + speculative-decode draft acceptance. Use after changing model config (cache_size, max_batch_size, draft/MTP, quant) or backend code, to confirm the server loads and generates correctly in one shot instead of restarting and eyeballing logs by hand.
---

# tabby-reload-verify

Automates the restart → wait-for-load → smoke-test → read-metrics loop. The draft
acceptance ("Draft: N accepted, M rejected, acceptance 0.xx") is printed to the
server's stdout, NOT returned by the API — so the server must be launched with its
output captured. This skill launches it in the background so the log is readable.

## Procedure

1. **Stop the currently running server** (frees the port and VRAM). Pick by how it
   was started:
   - **If a previous run of this skill launched it** via the Bash tool's
     `run_in_background`: stop that background task (TaskStop) — clean and exact.
   - **If the user launched it in their own terminal**: ask them to stop it
     (Ctrl+C). On Windows, reliably finding/killing a foreign server process is
     flaky — `netstat`/`psutil` may not report the `:5000` owner without admin,
     and the python process often shows an opaque `python.exe -` cmdline.
   - **Best-effort programmatic fallback** (verify it actually stopped afterward):
     ```bash
     netstat -ano | findstr :5000      # grab the PID, then: taskkill /F /PID <pid>
     ```
   Confirm the port is free before relaunching:
   `curl -s -m3 http://127.0.0.1:5000/health` should fail to connect (a 503 means
   it's still up but with no model). Also check
   `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.

2. **Launch the server in the background** with the user's usual start command,
   using the Bash tool's `run_in_background: true` so stdout is captured to a file.
   Note the output file path the tool returns — that's the server log. Example
   command (reuse the user's actual args):
   ```
   ./start.bat --model-dir <DIR> --model-name <MODEL> --backend exllamav3 \
     --max-seq-len 131072 --cache-mode Q4 --tool-format qwen3_coder
   ```

3. **Wait for ready:** poll the captured log (Read it) until you see
   `Model successfully loaded` AND `Uvicorn running`, or bail on a `Traceback` /
   `Insufficient VRAM` (OOM → see the `exl3-vram-fit` skill to find a fitting
   config).

4. **Smoke-test + read metrics** — pass the captured log path so the script can
   extract the Metrics/Draft line the server prints after the request:
   ```bash
   <venv-python> .claude/skills/tabby-reload-verify/verify.py \
     --repo-root . --server-log "<captured-log-path>" \
     --prompt "def quicksort(arr):\n" --max-tokens 200
   ```
   It reads the API key from `api_tokens.yml`, sends one completion, and prints
   the latest `Metrics (...)` + `Draft:` lines (Generate T/s + acceptance).

5. **Report**: loaded ✓/✗, generation ✓/✗, Generate T/s, draft acceptance. A
   healthy MTP run shows acceptance well above 0 (e.g. ~0.5–0.7 on code).

## Notes

- `--server-log` is essential for the draft-acceptance readout; without it the
  script can only confirm the generation succeeded (the API returns `usage: null`
  here, so no token count from the API).
- If verifying an already-running server you launched separately, point
  `--server-log` at wherever you redirected its stdout.
- To find a config that actually fits before restarting, use `exl3-vram-fit`.

#!/usr/bin/env python
"""
tabby-reload-verify — smoke-test a running tabbyAPI server and report the
speculative-decode draft acceptance.

Sends one generation, then (if given the server's stdout log) extracts the
"Metrics (ID ...)" line the server prints after the request — that line carries
Generate T/s and "Draft: N accepted, M rejected, acceptance 0.xx", which is NOT
returned by the API itself.

The API key is read from api_tokens.yml (api_key) unless --api-key is given.
"""
import argparse, json, re, sys, time, urllib.request, urllib.error, pathlib


def read_api_key(repo_root):
    p = pathlib.Path(repo_root) / "api_tokens.yml"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        m = re.match(r"\s*api_key\s*:\s*[\"']?([0-9a-zA-Z]+)", line)
        if m:
            return m.group(1)
    return None


def post(url, key, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:5000")
    ap.add_argument("--model", default=None, help="model name (optional; server uses loaded model)")
    ap.add_argument("--prompt", default="def quicksort(arr):\n")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--server-log", default=None,
                    help="path to the server's captured stdout (to read the Metrics line)")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    key = args.api_key or read_api_key(args.repo_root)
    if not key:
        print("No API key (pass --api-key or ensure api_tokens.yml exists).", file=sys.stderr)
        sys.exit(1)

    # health
    try:
        urllib.request.urlopen(args.base_url + "/health", timeout=5).read()
    except urllib.error.HTTPError as e:
        if e.code == 503:
            print(f"Server is up at {args.base_url} but reports 503 — no model loaded "
                  f"(or still loading). Cannot generate yet.", file=sys.stderr)
        else:
            print(f"Server unhealthy at {args.base_url}: HTTP {e.code}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Server not reachable at {args.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    log_pos = 0
    if args.server_log and pathlib.Path(args.server_log).exists():
        log_pos = pathlib.Path(args.server_log).stat().st_size  # only read NEW log after this

    body = {"prompt": args.prompt, "max_tokens": args.max_tokens, "temperature": 0}
    if args.model:
        body["model"] = args.model
    t0 = time.time()
    d = post(args.base_url + "/v1/completions", key, body, args.timeout)
    dt = time.time() - t0
    ch = (d.get("choices") or [{}])[0]
    usage = d.get("usage") or {}
    ct = usage.get("completion_tokens")
    print(f"generation OK: {dt:.2f}s, finish={ch.get('finish_reason')}, "
          f"completion_tokens={ct if ct is not None else 'n/a (usage null)'}")
    print("--- sample ---")
    print((ch.get("text") or "")[:240])

    # pull the Metrics line from the server log (printed after the request)
    if args.server_log and pathlib.Path(args.server_log).exists():
        time.sleep(0.7)  # let the server flush its metrics log line
        txt = pathlib.Path(args.server_log).read_text(errors="ignore")[log_pos:]
        metrics = [ln for ln in txt.splitlines() if "Metrics (ID" in ln]
        draft = [ln for ln in txt.splitlines() if "Draft:" in ln]
        if metrics:
            print("--- server metrics ---")
            print(metrics[-1].strip())
        if draft:
            print(draft[-1].strip())
        elif metrics:
            print("(no 'Draft:' line — draft/MTP not active for this request?)")
        if not metrics:
            print("(no Metrics line found in server log yet; check the log path/redirect)")


if __name__ == "__main__":
    main()

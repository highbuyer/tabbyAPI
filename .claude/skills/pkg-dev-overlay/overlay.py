#!/usr/bin/env python
"""
pkg-dev-overlay — try a dev-branch feature of an installed package WITHOUT
compiling, by overlaying the branch's pure-Python changes onto the installed
wheel and keeping the existing compiled extension (.pyd/.so).

Safe only when the compiled C/CUDA sources are unchanged between the installed
version and the target branch (or the changes don't affect code paths you use).
The tool diffs them and WARNS; you decide.

Modes:
  --dry-run   list changed files (py to overlay; C/CUDA flagged) — no writes
  --apply     backup originals, download branch .py files into the package,
              clear __pycache__, write a revert manifest
  --revert    restore originals from the manifest and delete added files

GitHub access honors https_proxy/http_proxy env vars (urllib).
"""
import argparse, json, os, sys, urllib.request, shutil, pathlib

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
C_EXT = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp")
MANIFEST = "overlay_manifest.json"


def http_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "pkg-dev-overlay"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def http_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pkg-dev-overlay"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def list_changed(repo, base, target):
    """Return (files, target_sha). files: list of dict(filename,status)."""
    d = http_json(f"{API}/repos/{repo}/compare/{base}...{target}")
    return d.get("files", []), d.get("commits", [{}])[-1].get("sha")


def categorize(files, prefix):
    py, cext, other = [], [], []
    for f in files:
        fn = f["filename"]
        if not fn.startswith(prefix):
            continue  # outside the installed package (eval/, util/, etc.)
        rel = fn[len(prefix):]
        if fn.endswith(".py"):
            py.append((rel, fn, f["status"]))
        elif fn.endswith(C_EXT):
            cext.append((rel, fn, f["status"]))
        else:
            other.append((rel, fn, f["status"]))
    return py, cext, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name, e.g. turboderp-org/exllamav3")
    ap.add_argument("--base-ref", required=True, help="installed version tag, e.g. v0.0.40")
    ap.add_argument("--target-ref", required=True, help="branch/sha to overlay, e.g. dev")
    ap.add_argument("--pkg-dir", required=True, help="installed package dir (.../site-packages/<pkg>)")
    ap.add_argument("--repo-prefix", required=True,
                    help="path prefix in the repo that maps to pkg-dir, e.g. 'exllamav3/'")
    ap.add_argument("--backup-dir", default=None, help="default: <pkg-dir>/../_overlay_backup_<pkg>")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="apply even if C/CUDA sources changed (you accept the risk)")
    args = ap.parse_args()

    pkg = pathlib.Path(args.pkg_dir).resolve()
    prefix = args.repo_prefix if args.repo_prefix.endswith("/") else args.repo_prefix + "/"
    backup = pathlib.Path(args.backup_dir).resolve() if args.backup_dir \
        else pkg.parent / f"_overlay_backup_{pkg.name}"

    if args.revert:
        man = backup / MANIFEST
        if not man.exists():
            print(f"No manifest at {man}; nothing to revert.", file=sys.stderr)
            sys.exit(1)
        m = json.loads(man.read_text())
        for rel in m["overlaid"]:
            src = backup / rel
            if src.exists():
                shutil.copy2(src, pkg / rel)
                print(f"  restored {rel}")
        for rel in m["added"]:
            p = pkg / rel
            if p.exists():
                p.unlink()
                print(f"  removed  {rel}")
        # clear bytecode
        for pc in pkg.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
        print(f"Reverted overlay {m['target_ref']} -> restored {m['base_ref']}.")
        return

    files, target_sha = list_changed(args.repo, args.base_ref, args.target_ref)
    py, cext, other = categorize(files, prefix)

    print(f"repo={args.repo}  {args.base_ref}...{args.target_ref}  (target sha {target_sha})")
    print(f"package files changed: {len(py)} python, {len(cext)} C/CUDA, {len(other)} other")
    if cext:
        print("\n  !! C/CUDA sources changed (compiled extension is NOT rebuilt by this tool):")
        for rel, fn, st in cext:
            print(f"     {st:9} {fn}")
        print("  -> overlay is safe ONLY if these don't affect code paths you use")
        print("     (e.g. multi-GPU/TP kernels on a single-GPU box). Review before --apply.")
    print("\n  python files to overlay:")
    for rel, fn, st in py:
        print(f"     {st:9} {rel}")

    if args.dry_run:
        print("\n(dry-run: no changes written)")
        return

    if cext and not args.force:
        print("\nRefusing to --apply because C/CUDA sources changed. Re-run with --force "
              "if you've confirmed they don't matter for your use.", file=sys.stderr)
        sys.exit(2)

    backup.mkdir(parents=True, exist_ok=True)
    overlaid, added = [], []
    for rel, fn, st in py:
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        existed = dst.exists()
        if existed:
            bdst = backup / rel
            bdst.parent.mkdir(parents=True, exist_ok=True)
            if not bdst.exists():
                shutil.copy2(dst, bdst)
            overlaid.append(rel)
        else:
            added.append(rel)
        data = http_bytes(f"{RAW}/{args.repo}/{target_sha}/{fn}")
        dst.write_bytes(data)
        print(f"  {'overlaid' if existed else 'added   '} {rel} ({len(data)} bytes)")

    for pc in pkg.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)

    (backup / MANIFEST).write_text(json.dumps({
        "repo": args.repo, "base_ref": args.base_ref, "target_ref": args.target_ref,
        "target_sha": target_sha, "overlaid": overlaid, "added": added,
    }, indent=2))
    print(f"\nApplied {len(overlaid)+len(added)} files. Backup + manifest at {backup}")
    print(f"Verify:  python -c \"import {pkg.name}\"   then test your feature.")
    print(f"Revert:  this script with --revert (same --pkg-dir/--backup-dir).")


if __name__ == "__main__":
    main()

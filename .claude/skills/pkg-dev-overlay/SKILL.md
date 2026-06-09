---
name: pkg-dev-overlay
description: Try a dev-branch feature of an installed Python package WITHOUT compiling, by overlaying the branch's pure-Python changes onto the installed wheel and keeping the existing compiled extension (.pyd/.so). Use when a feature exists only on a git branch (not a released wheel), building from source is slow or blocked (e.g. CUDA toolkit/torch version mismatch), and the branch's changes are mostly Python. Diffs the C/CUDA sources first and warns; backs up originals; supports one-command revert.
---

# pkg-dev-overlay

Get a branch-only feature running in minutes instead of a full source build, when
the C/CUDA part of the package is unchanged (or its changes don't touch your code
paths). This is exactly how exllamav3 dev MTP was put on the installed 0.0.40
wheel in this repo. See project memory `mtp-exl3-setup`.

## When this is safe (read first)

Overlaying Python while keeping the OLD compiled extension is safe **only if** the
branch's compiled C/CUDA sources are unchanged, OR the changed ones are on code
paths you never hit (e.g. multi-GPU/tensor-parallel kernels on a single-GPU box),
AND the pybind/binding file didn't add new exported symbols the new Python calls.
The tool surfaces C/CUDA changes so you can judge; it refuses `--apply` if any
changed unless you pass `--force`.

If the C/CUDA changes DO matter, don't overlay — build from source instead.

## Procedure

1. **Find the refs:** the installed version tag (`--base-ref`, e.g. `v0.0.40` —
   check `importlib.metadata.version(pkg)`) and the branch/sha to try
   (`--target-ref`, e.g. `dev`). Find the installed package dir:
   ```bash
   <venv-python> -c "import <pkg>,os; print(os.path.dirname(<pkg>.__file__))"
   ```
   `--repo-prefix` is the path inside the repo that maps to that dir (usually
   `<pkg>/`, e.g. `exllamav3/`).

2. **Dry-run** to see what changes and whether C/CUDA is touched (honors
   `https_proxy`/`http_proxy`):
   ```bash
   https_proxy=$PROXY <venv-python> .claude/skills/pkg-dev-overlay/overlay.py \
     --repo turboderp-org/exllamav3 --base-ref v0.0.40 --target-ref dev \
     --pkg-dir <site-packages>/exllamav3 --repo-prefix exllamav3/ --dry-run
   ```
   Review the C/CUDA list. Decide if overlay is safe (see above). If unsure,
   inspect those files' diffs on GitHub before proceeding.

3. **Apply** (backs up originals, writes a revert manifest, clears `__pycache__`):
   ```bash
   ... --apply          # add --force only if C/CUDA changed and you accept it
   ```

4. **Verify:** `import <pkg>` works, then exercise the new feature
   (e.g. construct the new class / config component). Pin the target sha somewhere
   (the manifest records it).

5. **Revert** any time (same `--pkg-dir`/`--backup-dir`):
   ```bash
   ... --revert
   ```

## Notes

- Reinstalling/upgrading the wheel wipes the overlay (it lives in site-packages).
  Re-apply after any `pip install`.
- Files outside `--repo-prefix` (tests, eval/, util/) are ignored — they aren't
  part of the installed package.
- The backup dir + `overlay_manifest.json` default to
  `<pkg-dir>/../_overlay_backup_<pkg>`. Keep it; revert needs it.

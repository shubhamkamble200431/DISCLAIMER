"""Build refusal_study.zip -- the shippable bundle for the DGX (README_STUDY.md section 9).

WHAT: code + configs + SLURM launchers + data/processed for BOTH subprojects, wrapped in a
top-level `refusal/` directory so it unpacks with the same shape as refusal_wide.zip.

WHY A PYTHON SCRIPT AND NOT A .sh:
  * This is built on Windows. `zip` is not on PATH under PowerShell, and Compress-Archive
    writes backslash separators into the member names, which unzip on Linux turns into
    files literally called "refusal_generation\\src\\steer.py". zipfile writes forward
    slashes on every platform.
  * The exclusions are the whole point of the bundle (no weights, no activations). A shell
    glob list is a blocklist by accident; here every candidate is matched against an
    explicit allowlist AND then re-checked against a deny list, so a 567 MiB best_state.pt
    or a 178k-file hidden/ tree cannot slip in even if someone widens an include pattern.
  * --verify needs a per-member content hash, which needs a real language.

WHAT IS DELIBERATELY NOT IN HERE (by default):
  * model weights (*.pt/*.bin/*.safetensors) -- including C*'s best_state.pt. The DGX either
    already has the run dir or it is shipped separately; see the corrupt-archive note below.
    OPT-IN: --with-checkpoint adds exactly the 8 files of the frozen C* run dir
    (longformer_42_20260916-160022) as an explicitly enumerated set. That set bypasses the
    allowlist/deny machinery on purpose -- it is named file by file in CKPT_FILES, so
    widening an include glob still cannot pull in a weight, and the only weight that can
    ever ship is the one C* checkpoint the study is frozen against. val_preds.npy and
    val_probs.npy are part of that set and are NOT optional: every scoring stage gates on
    val parity against them, so a checkpoint shipped without them fails at run time.
  * activations (data/generated/*/hidden/**, *.npz) -- 178k files, hundreds of GB.
  * answers.jsonl from the OLD roster (gemma-4-e2b, qwen3-5-4b, qwen3-5-9b). Those are
    phase-1 outputs; step 1 regenerates from the new roster on the DGX.
  * data/study/*.json -- the RETIRED study_splits.json / selected_layers.json were derived
    from the discarded roster. Shipping them invites someone to reuse a frozen hash that no
    longer corresponds to the models being run.

NOTE ON longformer_ckpt.zip: D:/IITIndore/refusal/longformer_ckpt.zip is CORRUPT (unzip
reports no end-of-central-directory -- a truncated upload). It is not touched by this
script. A valid copy of the run dir is in the working tree; re-zip and `unzip -t` it, or
confirm the DGX already holds it. See README_STUDY.md section 9.

Usage:
  python -X utf8 scripts/make_study_bundle.py
  python -X utf8 scripts/make_study_bundle.py --dry-run
  python -X utf8 scripts/make_study_bundle.py --verify
  python -X utf8 scripts/make_study_bundle.py --with-checkpoint \
      --out D:/IITIndore/refusal/refusal_study_full.zip
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # refusal_generation
INTERN = ROOT.parent                                    # refusal_intern
REPO = INTERN.parent                                    # D:/IITIndore/refusal
GEN = ROOT
CLF = INTERN / "refusal_classifier"

DEFAULT_OUT = REPO / "refusal_study.zip"
ARC_PREFIX = "refusal"                                  # matches refusal_wide.zip's shape

# --------------------------------------------------------------------------- #
# what goes in -- (source dir, glob, recursive) relative to a subproject root
# --------------------------------------------------------------------------- #
# Allowlist. Anything not matched here is simply absent; there is no "everything
# else" fallback, which is what keeps weights and activations out by construction.
GEN_INCLUDE = [
    ("", "README_STUDY.md", False),
    ("", "MANUAL.md", False),
    ("", "SCREEN_MANUAL.md", False),
    ("", "DEPLOY_README.md", False),
    ("configs", "*.yaml", False),
    ("src", "*.py", False),
    ("scripts", "*.py", False),
    ("scripts", "*.sh", False),
    ("scripts/slurm", "*.sh", False),
    ("scripts/slurm", "*.slurm", False),
    ("data/processed", "*.parquet", False),
]

# The scoring stages import src.train / src.metrics / src.data and
# scripts/predict.py from the classifier repo, so its code ships too -- the
# bundle has to be able to run BOTH job families.
CLF_INCLUDE = [
    ("", "README*.md", False),
    ("", "requirements.txt", False),
    ("configs", "*.yaml", False),
    ("src", "*.py", False),
    ("scripts", "*.py", False),
    ("scripts", "*.sh", False),
    ("scripts/slurm", "*.sh", False),
    ("scripts/slurm", "*.slurm", False),
    ("data/processed", "*.parquet", False),
]

# Repo-root files worth carrying (plan.md is the spec every script cites).
REPO_INCLUDE = [
    ("", "requirements.txt", False),
    ("", "plan.md", False),
]

# Files lifted to the TOP of the unpacked tree, i.e. refusal/<name>, regardless of
# where they live in the working tree. The archive is rooted at a single `refusal/`
# segment (that is what makes it unzip cleanly inside /raid/chandresh/KPdubey/), so
# "top level" means refusal/<name> -- putting a member outside refusal/ would create a
# second top-level segment and break the unpack shape. DEPLOY_README.md is the first
# thing a reader should see, and it also ships in its home directory via GEN_INCLUDE.
EXTRA_TOP_LEVEL = [
    (GEN / "DEPLOY_README.md", "DEPLOY_README.md"),
]

# --------------------------------------------------------------------------- #
# the frozen C* checkpoint -- opt-in via --with-checkpoint
# --------------------------------------------------------------------------- #
# Enumerated file by file on purpose. This is the ONE weight the bundle may carry,
# and it is not reachable through the allowlist: `runs` is a denied path component,
# *.pt and *.npy are denied patterns, and best_state.pt is 70x MAX_MEMBER_BYTES.
# An explicit list means no future include-glob edit can ever add a second weight.
CKPT_RUN = (CLF / "results" / "runs" / "longformer_42_20260916-160022")
CKPT_FILES = [
    "best_state.pt",
    "config_snapshot.json",
    "metrics.json",
    "train.log",
    "val_preds.npy",
    "val_probs.npy",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]
# Without these two the scoring stages cannot run their val-parity gate, so a
# checkpoint missing them is worse than no checkpoint at all -- it fails late.
CKPT_REQUIRED = ["val_preds.npy", "val_probs.npy", "best_state.pt"]

# Large members get their own compresslevel; code stays at 9.
#
# MEASURED on best_state.pt (594,738,530 B, this machine, 2026-09-21) -- do not
# guess at these, the intuition "torch files are incompressible so use level 1" is
# WRONG on both halves:
#     STORED      594.7 MB  100.00%    0.3s
#     DEFLATE-1   613.4 MB  103.14%    5.0s   <-- EXPANDS the file by 18 MB
#     DEFLATE-6   531.2 MB   89.32%   13.0s
#     DEFLATE-9   526.9 MB   88.59%   15.4s
# The file is ~11% compressible (fp32 tensors have low-entropy exponent bytes), and
# deflate-1's fast path emits worse-than-stored output on it. So level 1 is banned
# below; 6 is the default -- the point of a "low" level was to not burn minutes, and
# 13s is not minutes. Raise to 9 for ~4 MB more, or drop to 0 (STORED) for an
# instant build, via --big-compresslevel.
LOW_COMPRESS_BYTES = 32 * 1024 * 1024
LOW_COMPRESS_LEVEL = 6

# Deny list, applied to every candidate AFTER the allowlist. Belt and braces:
# if someone later widens an include glob, this still refuses to ship weights.
DENY_PATTERNS = [
    "*.pt", "*.bin", "*.safetensors", "*.ckpt", "*.h5", "*.onnx",
    "*.npz", "*.npy", "*.zip", "*.tar", "*.tar.gz", "*.gz",
    "*.pyc", "*.log",
]
DENY_PATH_PARTS = {
    "__pycache__", ".venv", ".venv-gen", ".hf_cache", ".git",
    "hidden", "generated", "runs", "models", "logs",
}
# A single file must never exceed this in a code bundle; tripping it means a
# weight or an activation shard got through and the build aborts loudly.
MAX_MEMBER_BYTES = 8 * 1024 * 1024


class Tee:
    """stdout + <out_dir>/<stage>.log, the house logger."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def denied(rel: str, size: int) -> str | None:
    """Return the reason this path is refused, or None if it may ship."""
    parts = set(Path(rel).parts)
    hit = parts & DENY_PATH_PARTS
    if hit:
        return f"path component {sorted(hit)[0]!r}"
    name = Path(rel).name
    for pat in DENY_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return f"pattern {pat}"
    if size > MAX_MEMBER_BYTES:
        return f"size {size / 1e6:.1f} MB > {MAX_MEMBER_BYTES / 1e6:.0f} MB cap"
    return None


def collect(base: Path, spec: list[tuple[str, str, bool]], arc_base: str,
            log) -> list[tuple[Path, str]]:
    """Resolve the allowlist against the filesystem. Sorted => deterministic order."""
    out: dict[str, Path] = {}
    for sub, pat, recursive in spec:
        d = base / sub if sub else base
        if not d.is_dir():
            log(f"  .. absent, skipped: {d}")
            continue
        it = d.rglob(pat) if recursive else d.glob(pat)
        for p in it:
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            arc = f"{ARC_PREFIX}/{arc_base}/{rel}" if arc_base else f"{ARC_PREFIX}/{rel}"
            why = denied(rel, p.stat().st_size)
            if why:
                # Not a warning to ignore: the allowlist matched something the
                # deny list refuses, which means one of the two is wrong.
                log(f"  !! EXCLUDED {rel}  ({why})")
                continue
            out[arc] = p
    return sorted(out.items(), key=lambda kv: kv[0])  # type: ignore[return-value]


def collect_extra(log) -> list[tuple[str, Path]]:
    """Named single files promoted to refusal/<name>. Deny list still applies."""
    out = []
    for src, name in EXTRA_TOP_LEVEL:
        if not src.is_file():
            log(f"  .. absent, skipped: {src}")
            continue
        why = denied(name, src.stat().st_size)
        if why:
            log(f"  !! EXCLUDED {name}  ({why})")
            continue
        out.append((f"{ARC_PREFIX}/{name}", src))
    return sorted(out, key=lambda kv: kv[0])


def collect_checkpoint(log) -> list[tuple[str, Path]]:
    """The frozen C* run dir, by explicit name. Bypasses denied() by design.

    Aborts the build rather than shipping a half checkpoint: a missing
    val_preds.npy only surfaces on the DGX, hours into a scoring job.
    """
    if not CKPT_RUN.is_dir():
        log(f"!! --with-checkpoint but run dir is absent: {CKPT_RUN}")
        sys.exit(1)
    arc_base = f"{ARC_PREFIX}/refusal_classifier/" \
               f"{CKPT_RUN.relative_to(CLF).as_posix()}"
    out, missing = [], []
    for rel in CKPT_FILES:
        p = CKPT_RUN / rel
        if not p.is_file():
            missing.append(rel)
            continue
        out.append((f"{arc_base}/{rel}", p))
    for rel in missing:
        lvl = "!! MISSING REQUIRED" if rel in CKPT_REQUIRED else "!! missing"
        log(f"{lvl} checkpoint file {rel}")
    if missing:
        log("!! refusing to ship an incomplete checkpoint -- the scoring stages "
            "gate on val parity and would fail at run time")
        sys.exit(1)
    return sorted(out, key=lambda kv: kv[0])


def build_inventory(log, with_checkpoint: bool = False) -> list[tuple[str, Path]]:
    log("collecting refusal_generation ...")
    items = collect(GEN, GEN_INCLUDE, "refusal_generation", log)
    log("collecting refusal_classifier ...")
    items += collect(CLF, CLF_INCLUDE, "refusal_classifier", log)
    log("collecting repo root ...")
    items += collect(REPO, REPO_INCLUDE, "", log)
    log("collecting top-level extras ...")
    items += collect_extra(log)
    if with_checkpoint:
        log(f"collecting C* checkpoint ({len(CKPT_FILES)} files) ...")
        items += collect_checkpoint(log)
    # (arc, path) pairs, arc-sorted within each group; sort once more globally
    return sorted(((a, p) for a, p in items), key=lambda kv: kv[0])


def required_present(items: list[tuple[str, Path]], log) -> bool:
    """Fail loudly if a file the study cannot run without is missing."""
    must = [
        "refusal/refusal_generation/src/steer.py",
        "refusal/refusal_generation/src/generate.py",
        "refusal/refusal_generation/configs/study.yaml",
        "refusal/refusal_generation/README_STUDY.md",
        "refusal/DEPLOY_README.md",
        "refusal/refusal_generation/DEPLOY_README.md",
        "refusal/refusal_generation/scripts/step1_generate_label_split.py",
        "refusal/refusal_generation/scripts/step2_directions_steering.py",
        "refusal/refusal_generation/scripts/step3_geometry.py",
        "refusal/refusal_generation/scripts/step4_ood_mozilla.py",
        "refusal/refusal_generation/scripts/slurm/study_config.sh",
        "refusal/refusal_generation/scripts/slurm/run_study.sh",
        "refusal/refusal_generation/scripts/slurm/23_step2_steer_dgx.slurm",
        "refusal/refusal_classifier/scripts/predict.py",
        "refusal/refusal_classifier/src/train.py",
        "refusal/refusal_classifier/src/metrics.py",
        "refusal/refusal_classifier/configs/longformer.yaml",
    ]
    have = {a for a, _ in items}
    missing = [m for m in must if m not in have]
    for m in missing:
        log(f"!! MISSING REQUIRED MEMBER {m}")
    return not missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"archive path (default {DEFAULT_OUT.name}); use e.g. "
                         f"refusal_study_full.zip for the combined build")
    ap.add_argument("--with-checkpoint", action="store_true",
                    help="also ship the frozen C* run dir "
                         f"({CKPT_RUN.name}, {len(CKPT_FILES)} files, ~571 MB). "
                         "Default OFF: the code-only bundle stays ~3.8 MB.")
    ap.add_argument("--big-compresslevel", type=int, default=LOW_COMPRESS_LEVEL,
                    choices=[0, 2, 3, 4, 5, 6, 7, 8, 9],
                    help=f"compresslevel for members >= "
                         f"{LOW_COMPRESS_BYTES // (1024 * 1024)} MB "
                         f"(default {LOW_COMPRESS_LEVEL}; 0 means ZIP_STORED). "
                         "Level 1 is deliberately not offered: measured, it "
                         "EXPANDS best_state.pt by 3%%.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the manifest and sizes, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="re-test an existing archive against its manifest; "
                         "exit non-zero on drift")
    args = ap.parse_args()

    out: Path = args.out
    manifest_path = out.with_suffix(".manifest.json")
    log = Tee(out.parent / "make_study_bundle.log")

    try:
        # ---------------- verify ------------------------------------------- #
        if args.verify:
            if not out.exists() or not manifest_path.exists():
                log(f"!! missing {out if not out.exists() else manifest_path}")
                sys.exit(1)
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            ok = True

            with zipfile.ZipFile(out) as zf:
                bad = zf.testzip()
                if bad is not None:
                    log(f"!! CRC failure in {bad}")
                    ok = False
                names = sorted(zf.namelist())
                recorded = sorted(m["arcname"] for m in man["members"])
                if names != recorded:
                    log(f"!! member list drift: archive {len(names)} vs "
                        f"manifest {len(recorded)}")
                    for n in set(names) ^ set(recorded):
                        log(f"   differs: {n}")
                    ok = False
                else:
                    for m in man["members"]:
                        got = hashlib.sha256(zf.read(m["arcname"])).hexdigest()
                        if got != m["sha256"]:
                            log(f"!! content drift: {m['arcname']}")
                            ok = False

            got_zip = sha256_file(out)
            if got_zip != man["archive_sha256"]:
                # Not fatal on its own: zip timestamps make the container bytes
                # non-reproducible. Member hashes above are the real check.
                log(f".. archive sha256 differs (committed {man['archive_sha256'][:16]} "
                    f"recomputed {got_zip[:16]}) -- container bytes only")

            log(f"verify: {'PASS' if ok else 'FAIL'}  {out}")
            sys.exit(0 if ok else 1)

        # ---------------- build -------------------------------------------- #
        log(f"repo  {REPO}")
        log(f"out   {out}")
        log(f"ckpt  {'INCLUDED -- ' + str(CKPT_RUN) if args.with_checkpoint else 'excluded (default)'}")
        items = build_inventory(log, with_checkpoint=args.with_checkpoint)
        if not required_present(items, log):
            log("!! refusing to build an incomplete bundle")
            sys.exit(1)

        # Shell and SLURM files MUST be LF. This box is Windows, where Python's text-mode
        # write_text() silently turns every \n into \r\n: sbatch then rejects the script
        # ("Batch script contains DOS line breaks") and bash dies on `$'\r': command not
        # found` -- and run_study.sh is called by every job, so one CRLF file breaks the
        # whole study. It shipped once already; check bytes, not what grep reports.
        crlf = [a for a, p in items
                if a.endswith((".sh", ".slurm")) and b"\r" in p.read_bytes()]
        if crlf:
            for a in crlf:
                log(f"!! CRLF line endings in {a}")
            log("!! refusing to build: normalise with  sed -i 's/\\r$//' <file>  "
                "(or write the bytes with \\n) and rebuild")
            sys.exit(1)

        total = sum(p.stat().st_size for _, p in items)
        log("")
        log(f"{'size':>10}  arcname")
        log(f"{'-' * 10}  {'-' * 60}")
        for arc, p in items:
            log(f"{p.stat().st_size:>10}  {arc}")
        log(f"{'-' * 10}  {'-' * 60}")
        log(f"{total:>10}  {len(items)} files, {total / 1e6:.2f} MB uncompressed")

        if args.dry_run:
            log("\n--dry-run: nothing written")
            return

        members = []
        t_build = time.perf_counter()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for arc, p in items:
                nbytes = p.stat().st_size
                big = nbytes >= LOW_COMPRESS_BYTES
                lvl = args.big_compresslevel if big else 9
                ctype = (zipfile.ZIP_STORED if (big and lvl == 0)
                         else zipfile.ZIP_DEFLATED)
                t0 = time.perf_counter()
                zf.write(p, arcname=arc, compress_type=ctype,
                         compresslevel=(None if ctype == zipfile.ZIP_STORED
                                        else lvl))
                dt = time.perf_counter() - t0
                how = ("ZIP_STORED" if ctype == zipfile.ZIP_STORED
                       else f"ZIP_DEFLATED compresslevel={lvl}")
                if big:
                    info = zf.getinfo(arc)
                    log(f"  .. {arc}: {nbytes / 1e6:.1f} MB -> "
                        f"{info.compress_size / 1e6:.1f} MB "
                        f"({100 * info.compress_size / nbytes:.1f}%) "
                        f"via {how} in {dt:.1f}s")
                members.append({
                    "arcname": arc,
                    "source": p.relative_to(REPO).as_posix(),
                    "bytes": nbytes,
                    "sha256": sha256_file(p),
                    "compression": how,
                })
        log(f"  .. packed {len(items)} members in "
            f"{time.perf_counter() - t_build:.1f}s")

        # unzip -t equivalent, in-process: CRC-check every member before anyone
        # ships this. A bundle that fails here is the longformer_ckpt.zip story.
        with zipfile.ZipFile(out) as zf:
            bad = zf.testzip()
            if bad is not None:
                log(f"!! integrity test FAILED on {bad}")
                sys.exit(1)
            names = zf.namelist()
        n_in_zip = len(names)
        if n_in_zip != len(items):
            log(f"!! wrote {len(items)} but archive holds {n_in_zip}")
            sys.exit(1)

        # The unpack shape IS the deployment contract: exactly one top-level
        # segment, so `unzip` inside /raid/chandresh/KPdubey/ lands the tree at
        # /raid/chandresh/KPdubey/refusal/ and nothing spills beside it.
        tops = sorted({n.split("/")[0] for n in names})
        if tops != [ARC_PREFIX]:
            log(f"!! top-level segments {tops}, expected ['{ARC_PREFIX}']")
            sys.exit(1)
        log(f"  .. top-level segments {tops}  (exactly one, as required)")

        arc_sha = sha256_file(out)
        ckpt_members = [m for m in members
                        if "/results/runs/" in m["arcname"]]
        manifest = {
            "seed": None,          # no randomness in this artifact; field kept for
                                   # uniformity with the other committed artifacts
            "built_from": REPO.as_posix(),
            "archive": out.name,
            "archive_bytes": out.stat().st_size,
            "archive_sha256": arc_sha,
            "n_members": len(items),
            "uncompressed_bytes": total,
            "top_level_segments": tops,
            "big_member_bytes": LOW_COMPRESS_BYTES,
            "big_compresslevel": args.big_compresslevel,
            # The DGX copy is verified against this block alone: sha256 each file
            # under the unpacked run dir and compare. No need to trust the zip.
            "with_checkpoint": bool(args.with_checkpoint),
            "checkpoint": ({
                "run_dir": CKPT_RUN.relative_to(REPO).as_posix(),
                "arc_dir": (f"{ARC_PREFIX}/refusal_classifier/"
                            f"{CKPT_RUN.relative_to(CLF).as_posix()}"),
                "n_files": len(ckpt_members),
                "bytes": sum(m["bytes"] for m in ckpt_members),
                "files": [
                    {"rel": m["arcname"].split("/results/runs/")[1]
                                        .split("/", 1)[1],
                     "arcname": m["arcname"],
                     "bytes": m["bytes"],
                     "sha256": m["sha256"]}
                    for m in ckpt_members
                ],
            } if args.with_checkpoint else None),
            "excludes": {
                "patterns": DENY_PATTERNS,
                "path_components": sorted(DENY_PATH_PARTS),
                "max_member_bytes": MAX_MEMBER_BYTES,
                "note": ("no activations, no phase-1 answers, no retired "
                         "data/study artifacts; "
                         + ("C* checkpoint included by --with-checkpoint "
                            "(enumerated, exempt from the deny list)"
                            if args.with_checkpoint else "no weights")),
            },
            "members": members,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        log("")
        log(f"integrity test  PASS ({n_in_zip} members, CRC verified)")
        log(f"archive         {out}  {out.stat().st_size / 1e6:.2f} MB "
            f"(compressed from {total / 1e6:.2f} MB)")
        log(f"sha256          {arc_sha}")
        log(f"manifest        {manifest_path}")
        log("")
        if args.with_checkpoint:
            cbytes = sum(m["bytes"] for m in ckpt_members)
            log(f"C* checkpoint   INCLUDED: {len(ckpt_members)} files, "
                f"{cbytes / 1e6:.1f} MB unpacked")
            for m in ckpt_members:
                log(f"  {m['bytes']:>10}  {m['sha256'][:16]}  "
                    f"{m['arcname'].split('/results/runs/')[1]}")
            log("Per-file sha256 for all of them is in the manifest under "
                "\"checkpoint\" --")
            log("sha256sum the unpacked run dir on the DGX and diff against it.")
        else:
            log("C* weights are NOT in this bundle. Confirm the DGX holds")
            log("  /raid/chandresh/KPdubey/refusal/refusal_classifier/results/runs/"
                "longformer_42_20260916-160022/")
            log("or rebuild with --with-checkpoint -- longformer_ckpt.zip at the repo")
            log("root is corrupt (no end-of-central-directory) and must not be used.")

        # ---------------- deployment hint ---------------------------------- #
        log("")
        log("deploy:")
        log(f"  scp {out.as_posix()} "
            f"<user>@<dgx>:/raid/chandresh/KPdubey/")
        log(f"  ssh <user>@<dgx> 'cd /raid/chandresh/KPdubey && "
            f"unzip -o {out.name}'")
        log(f"  -> lands at /raid/chandresh/KPdubey/{ARC_PREFIX}/ "
            f"(read {ARC_PREFIX}/DEPLOY_README.md first)")
    finally:
        log.close()


if __name__ == "__main__":
    main()

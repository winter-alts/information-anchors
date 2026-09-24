#!/usr/bin/env python3
"""Build reusable official MI sidecars and optionally run one full matrix cell.

Discovery is fitted once.  Query/candidate hidden states are then processed in
contiguous shards, and only compact score sidecars are passed to the official
Align-RAG/TS-RAG runners.  This keeps memory bounded for electricity while
preserving the exact flattened channel-major origin order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.downstream.gpu_policy import validate_device, validate_gpu_id

DEFAULT_METHODS = "base,official,uniform,random,low_mi,high_mi,mi_prior"
ATTRIBUTION_METHODS = "mi_student,student_no_mi,no_mi_prior,uniform_no_mi_prior,null_mi_prior,mi_prior_mmr,no_mi_prior_mmr,uniform_no_mi_prior_mmr,null_mi_prior_mmr"
ALIGN_V2_METHODS = (
    "base,official,uniform,random,low_mi,high_mi,mi_student,student_no_mi,"
    "no_mi_prior,uniform_no_mi_prior,null_mi_prior,mi_prior,mi_prior_mmr,"
    "no_mi_prior_mmr,uniform_no_mi_prior_mmr,null_mi_prior_mmr"
)
STUDENT_FREE_METHODS = frozenset({
    "base", "official", "uniform", "random", "low_mi", "high_mi",
    "high_mi_anchor", "low_mi_anchor", "random_patch", "all_patch",
    "mi_residual_prototype",
    "horizon_mi", "horizon_global_mi", "horizon_uniform", "horizon_random",
})


HORIZON_METHODS = frozenset({
    "horizon_mi", "horizon_global_mi", "horizon_uniform", "horizon_random"
})


def _validate_horizon_wrapper_request(args: argparse.Namespace) -> None:
    """Validate the immutable horizon protocol and, for test, its receipt."""
    requested = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    horizon_requested = [item for item in requested if item in HORIZON_METHODS]
    receipt_needed = bool(horizon_requested) or (
        str(args.split) == "test" and getattr(args, "selection_receipt", None) is not None
    )
    if not receipt_needed:
        return
    if horizon_requested:
        checks = (
            (bool(args.no_student), "--no-student is required"),
            (args.mi_target == "future_truth", "mi-target must be future_truth"),
            (args.mi_condition == "none", "mi-condition must be none"),
            (int(args.pool_k) == 20, "pool-k must be 20"),
            (int(args.top_k) == 10, "top-k must be 10"),
            (not bool(args.arm_bias), "arm-bias is not allowed"),
            (not bool(args.arm_attention_prior), "arm-attention-prior is not allowed"),
            (args.backend == "tsrag", "backend must be tsrag"),
            (args.gamma is None, "gamma override is not allowed"),
            (args.correction_gain_file is None, "correction-gain-file is not allowed"),
            (not bool(args.mi_blend_pool), "mi-blend-pool is not allowed"),
            (not bool(args.align_mi_v2), "align-mi-v2 is not allowed"),
            (not bool(args.selective_mi_gate), "selective-mi-gate is not allowed"),
        )
        violations = [message for passed, message in checks if not passed]
        if violations:
            raise ValueError("horizon MI wrapper protocol violation: " + "; ".join(violations))
    receipt_path = getattr(args, "selection_receipt", None)
    if receipt_path is None:
        if str(args.split) == "test" and horizon_requested:
            raise ValueError("horizon MI test request requires --selection-receipt")
        return
    path = Path(receipt_path).resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read selection receipt {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("selection receipt must be a JSON object")
    args.selection_receipt_path = str(path)
    args.selection_receipt_sha256 = hashlib.sha256(raw).hexdigest()
    if str(args.split) != "test":
        return
    if not bool(payload.get("test_unlocked", False)):
        raise ValueError("horizon MI selection was not admitted; test remains locked")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or str(args.dataset) not in datasets:
        raise ValueError(f"selection receipt has no dataset row for {args.dataset}")
    row = datasets[str(args.dataset)]
    if not isinstance(row, dict):
        raise ValueError(f"selection receipt row for {args.dataset} is malformed")
    selected_method = str(row.get("selected_method", ""))
    if horizon_requested:
        if selected_method != "horizon_mi":
            raise ValueError("horizon MI selection was not admitted; test remains locked to official")
        if requested != ["official", "horizon_mi"]:
            raise ValueError("admitted horizon MI test requires methods exactly official,horizon_mi")
    elif selected_method == "official" and requested != ["official"]:
        raise ValueError("official fallback test requires methods exactly official")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--query-shard-size", type=int, default=512)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="number of candidates passed to the frozen backend (must be <= pool-k)",
    )
    parser.add_argument(
        "--no-student", action="store_true",
        help="skip residual utility student fitting for the MI-Anchor-only protocol",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--discovery-per-channel", type=int, default=16)
    parser.add_argument("--student-oof-folds", type=int, default=5)
    parser.add_argument(
        "--student-target", choices=("residual_match", "align_future_mse"),
        default="residual_match",
    )
    parser.add_argument(
        "--projection-fit-samples", type=int, default=0,
        help="max train-discovery hidden rows used for PCA (0=all)",
    )
    parser.add_argument("--mi-condition", choices=("none", "recent"), default="none")
    parser.add_argument("--mi-target", choices=("residual", "future_truth"), default="residual",
                        help="MI target: forecast residual or future truth")
    parser.add_argument("--recent-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gamma", type=float, default=None,
                        help="override discovery-OOF MI rank-fusion gamma")
    parser.add_argument("--backend", choices=("none", "align", "tsrag"), default="none")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--attribution-controls", action="store_true",
                        help="fit equal-capacity no-MI/null-MI students in discovery state")
    parser.add_argument("--mmr-lambda", type=float, default=0.3,
                        help="history-only MI-prior MMR relevance/diversity tradeoff")
    parser.add_argument("--results-dir", default="")
    parser.add_argument(
        "--selection-receipt", default=None,
        help="validation admission receipt required before a horizon-MI test run",
    )
    parser.add_argument("--discovery-dir", default="",
                        help="reuse an existing train/discovery cache for another evaluation split")
    parser.add_argument("--pretrained-model-path", default="amazon/chronos-bolt-base")
    parser.add_argument("--retrieval-database-dir", default="repository_packages/external_rag/TS-RAG/retrieval_database")
    parser.add_argument("--retrieval-checkpoint", default="repository_packages/external_rag/TS-RAG/checkpoints/best.pth")
    parser.add_argument("--arm-bias", action="store_true",
                        help="pass MI-prior logits as an optional TS-RAG ARM bias")
    parser.add_argument("--arm-bias-strength", type=float, default=None,
                        help="override discovery-OOF ARM bias strength")
    parser.add_argument("--arm-bias-method", choices=("high_mi", "mi_prior", "both"), default="high_mi")
    parser.add_argument("--arm-attention-prior", action="store_true",
                        help="add a history-only MI prior to query-to-retrieved ARM attention")
    parser.add_argument(
        "--arm-attention-prior-method",
        choices=("high_mi", "mi_prior", "low_mi", "random", "uniform", "null_mi_prior"),
        default="high_mi",
    )
    parser.add_argument("--arm-attention-prior-strength", type=float, default=0.5)
    parser.add_argument("--save-preds", action="store_true",
                        help="ask the backend to persist predictions for paired bootstrap")
    parser.add_argument(
        "--mi-blend-pool", action="store_true",
        help="for Align-RAG, blend retrieved futures from each method's selected pool",
    )
    parser.add_argument(
        "--branch-mode", choices=("path_matched", "context_only", "future_only"),
        default=None,
        help="for Align-RAG, choose the MI context/future branch combination",
    )
    parser.add_argument(
        "--blend-score-eta", type=float, default=0.0,
        help="for Align-RAG, mix official and selector distances in the future branch",
    )
    parser.add_argument(
        "--align-mi-v2", action="store_true",
        help=("use the path-matched Align-MI v2 protocol: Align-aware student, "
              "MMR controls, and selected Top-10 future blending"),
    )
    parser.add_argument(
        "--selective-mi-gate", action="store_true",
        help="fit a discovery-only conformal gate and route MI-prior vs official per query",
    )
    parser.add_argument("--selective-gate-alpha", type=float, default=0.2)
    parser.add_argument("--selective-gate-fit-fraction", type=float, default=0.6)
    parser.add_argument("--selective-gate-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--selective-gate-top-k", type=int, default=4)
    parser.add_argument(
        "--correction-gain-file", default=None,
        help="validation-fitted Align correction gain JSON passed to the backend",
    )
    parser.add_argument("--metadata-frequency", required=True)
    parser.add_argument("--data-kind", choices=("ett_h_retrieve", "ett_m_retrieve", "custom_retrieve"), required=True)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--force-discovery", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="retain history hidden/input shards after score sidecars are built")
    args = parser.parse_args()
    args.gpu = validate_gpu_id(args.gpu)
    args.device = validate_device(args.device)
    if args.arm_attention_prior_strength < 0.0:
        raise ValueError("arm-attention-prior-strength must be non-negative")
    if args.align_mi_v2:
        if args.backend not in {"align", "none"}:
            raise ValueError("--align-mi-v2 requires --backend align or none")
        args.student_target = "align_future_mse"
        args.mi_blend_pool = True
        args.attribution_controls = True
        if args.methods == DEFAULT_METHODS:
            args.methods = ALIGN_V2_METHODS
    if args.attribution_controls and args.methods == DEFAULT_METHODS:
        args.methods = f"{DEFAULT_METHODS},{ATTRIBUTION_METHODS}"
    if args.selective_mi_gate:
        if args.no_student:
            raise ValueError("--selective-mi-gate requires the utility student")
        if args.student_target != "align_future_mse":
            raise ValueError(
                "--selective-mi-gate requires --student-target align_future_mse "
                "(or --align-mi-v2)"
            )
        requested = [item.strip() for item in args.methods.split(",") if item.strip()]
        if "selective_mi_prior" not in requested:
            requested.append("selective_mi_prior")
        args.methods = ",".join(requested)
    if args.pool_k < 1 or args.top_k < 1 or args.top_k > args.pool_k:
        raise ValueError("top-k must satisfy 1 <= top-k <= pool-k")
    requested_methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    if args.no_student and (args.attribution_controls or args.arm_bias
                            or not requested_methods.issubset(STUDENT_FREE_METHODS)):
        raise ValueError("--no-student only supports the four MI-Anchor controls without attribution students")
    if not 0.0 <= float(args.blend_score_eta) <= 1.0:
        raise ValueError("blend-score-eta must be in [0,1]")
    _validate_horizon_wrapper_request(args)
    return args


def _run(command: list[str], log: Path) -> dict[str, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=str(ROOT), text=True,
                            capture_output=True, check=False)
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return {"returncode": int(result.returncode),
            "status": "completed" if result.returncode == 0 else "failed",
            "log": str(log), "command": " ".join(command)}


def _official_window_count(args: argparse.Namespace) -> int:
    import importlib.util
    root = ROOT / "repository_packages/external_rag/align-rag"
    spec = importlib.util.spec_from_file_location("official_align_data_for_shards", root / "data.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import official Align-RAG data loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    windows = module.load(args.dataset, str(Path(args.data_dir).resolve()),
                          seq_len=512, pred_len=64, top_k=args.pool_k,
                          split=args.split)
    return len(windows)


def main() -> None:
    args = _args()
    horizon_requested = any(
        item.strip() in HORIZON_METHODS
        for item in str(args.methods).split(",")
        if item.strip()
    )
    if args.query_shard_size < 1:
        raise ValueError("query-shard-size must be positive")
    total = _official_window_count(args)
    lo0 = max(0, int(args.query_start))
    hi_total = total if args.query_end is None else min(total, int(args.query_end))
    if lo0 >= hi_total:
        raise ValueError("query range is empty")
    root = Path(args.output_root).resolve()
    external_discovery = bool(args.discovery_dir)
    discovery = Path(args.discovery_dir).resolve() if external_discovery else root / "discovery"
    inputs = root / "inputs"
    scores = root / "scores"
    work_root = root / "work"
    root.mkdir(parents=True, exist_ok=True)
    effective_branch_mode = args.branch_mode or (
        "path_matched" if args.mi_blend_pool else "context_only"
    )
    manifest: dict[str, object] = {
        "dataset": args.dataset, "split": args.split, "seq_len": 512, "pred_len": 64,
        "pool_k": args.pool_k, "top_k": args.top_k,
        "query_start": lo0, "query_end": hi_total,
        "query_windows": hi_total - lo0, "query_shard_size": args.query_shard_size,
        "backend": args.backend, "methods": args.methods,
        "mi_anchor_protocol": bool(args.no_student),
        "student_enabled": not args.no_student,
        "attribution_controls": bool(args.attribution_controls),
        "student_target": args.student_target,
        "arm_attention_prior": bool(args.arm_attention_prior),
        "arm_attention_prior_method": args.arm_attention_prior_method,
        "arm_attention_prior_strength": float(args.arm_attention_prior_strength),
        "mmr_lambda": float(args.mmr_lambda),
        "branch_mode": effective_branch_mode,
        "path_matched": effective_branch_mode == "path_matched",
        "blend_score_eta": float(args.blend_score_eta),
        "align_mi_v2": bool(args.align_mi_v2),
        "selective_gate_enabled": bool(args.selective_mi_gate),
        "selective_gate_alpha": float(args.selective_gate_alpha),
        "selective_gate_fit_fraction": float(args.selective_gate_fit_fraction),
        "selective_gate_ridge_alpha": float(args.selective_gate_ridge_alpha),
        "selective_gate_top_k": int(args.selective_gate_top_k),
        "correction_gain_file": str(Path(args.correction_gain_file).resolve())
        if args.correction_gain_file else None,
        "mi_target_source": args.mi_target,
        "horizon_mi_protocol": bool(horizon_requested),
        "selection_receipt_path": getattr(args, "selection_receipt_path", None),
        "selection_receipt_sha256": getattr(args, "selection_receipt_sha256", None),
        "discovery": str(discovery), "inputs": str(inputs), "scores": str(scores),
        "shards": [],
    }
    backend_results = (
        Path(args.results_dir).resolve()
        if args.results_dir else root / f"{args.backend}_results"
    )
    manifest["results_dir"] = str(backend_results) if args.backend != "none" else None
    discovery_summary = discovery / "summary.json"
    expected_mi_source = str(args.mi_target)
    if args.force_discovery or not discovery_summary.exists():
        discovery_command = [
            sys.executable, str(ROOT / "scripts/information_anchor/prepare_official_mi_discovery.py"),
            "--dataset", args.dataset, "--data-dir", str(Path(args.data_dir).resolve()),
            "--reference-config", str(Path(args.reference_config).resolve()),
            "--output-dir", str(discovery), "--pool-k", str(args.pool_k),
            "--batch-size", str(args.batch_size), "--discovery-per-channel", str(args.discovery_per_channel),
            "--student-oof-folds", str(args.student_oof_folds), "--mi-condition", args.mi_condition,
            "--student-target", args.student_target,
            "--mi-target", args.mi_target,
            "--recent-length", str(args.recent_length), "--seed", str(args.seed),
            "--projection-fit-samples", str(args.projection_fit_samples),
        ]
        if args.selective_mi_gate:
            discovery_command += [
                "--selective-mi-gate",
                "--selective-gate-alpha", str(args.selective_gate_alpha),
                "--selective-gate-fit-fraction", str(args.selective_gate_fit_fraction),
                "--selective-gate-ridge-alpha", str(args.selective_gate_ridge_alpha),
                "--selective-gate-top-k", str(args.selective_gate_top_k),
            ]
        if args.attribution_controls:
            discovery_command += ["--attribution-controls"]
        if args.no_student:
            discovery_command += ["--skip-student"]
        # Keep discovery and the backend runner on the same explicitly chosen
        # accelerator.  Previously ``--gpu`` only reached the final runner,
        # so large-channel discovery silently fell back to the config/default
        # device (or CPU) and could take hours.
        discovery_device = args.device or f"cuda:{args.gpu}"
        discovery_command += ["--device", discovery_device]
        manifest["discovery_run"] = _run(discovery_command, root / "logs/discovery.log")
        if manifest["discovery_run"]["status"] != "completed":
            Path(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            raise RuntimeError("discovery fitting failed; see manifest log")
    else:
        existing = json.loads(discovery_summary.read_text(encoding="utf-8"))
        existing_source = str(existing.get("mi_target_source", "residual"))
        existing_student = bool(existing.get("student_enabled", True))
        if args.no_student and existing_student:
            raise ValueError("--no-student requires a discovery cache built with --skip-student")
        requested_methods = {item.strip() for item in args.methods.split(",") if item.strip()}
        if (not args.no_student and not existing_student
                and not requested_methods.issubset(STUDENT_FREE_METHODS)):
            raise ValueError("discovery cache has no student state required by the requested methods")
        if existing_source != expected_mi_source:
            raise ValueError(
                f"discovery cache target={existing_source!r} does not match "
                f"requested --mi-target={expected_mi_source!r}; use a new output-root "
                "or --force-discovery"
            )
        existing_student_target = str(existing.get("student_target", "residual_match"))
        if existing_student_target != args.student_target:
            raise ValueError(
                f"discovery cache student_target={existing_student_target!r} does not match "
                f"requested --student-target={args.student_target!r}; use a new output-root "
                "or --force-discovery"
            )
        if args.attribution_controls and not bool(existing.get("attribution_controls", False)):
            raise ValueError(
                "cached discovery state lacks attribution controls; rerun with --force-discovery"
            )
        if args.selective_mi_gate and not bool(existing.get("selective_gate_enabled", False)):
            raise ValueError(
                "cached discovery state lacks selective gate; rerun with --force-discovery"
            )
        manifest["discovery_run"] = {"status": "reused", "summary": str(discovery_summary)}
    discovery_metadata = json.loads(discovery_summary.read_text(encoding="utf-8"))
    if horizon_requested:
        protocol_ok = (
            bool(discovery_metadata.get("horizon_mi_protocol", False))
            and str(discovery_metadata.get("mi_target_source")) == "future_truth"
            and int(discovery_metadata.get("horizon_block_size", -1)) == 16
            and int(discovery_metadata.get("horizon_block_count", -1)) == 4
        )
        if not protocol_ok:
            raise ValueError(
                "discovery cache predates the horizon MI protocol; use a fresh "
                "output root or --force-discovery"
            )
        state_path = discovery / "discovery_state.npz"
        if not state_path.exists():
            raise ValueError(
                "horizon MI discovery state is missing; use a fresh output root "
                "or --force-discovery"
            )
        with np.load(state_path, allow_pickle=False) as state:
            missing_profile = sorted({
                "horizon_mi_observed", "horizon_mi_weights",
                "global_mi_observed", "global_mi_weights",
            } - set(state.files))
        if missing_profile:
            raise ValueError(
                "horizon MI discovery state is missing "
                f"{missing_profile}; use a fresh output root or --force-discovery"
            )
    tuned = discovery_metadata.get("hyperparameters", {})
    gamma = float(args.gamma if args.gamma is not None else tuned.get("gamma", 0.7))
    arm_bias_strength = float(
        args.arm_bias_strength if args.arm_bias_strength is not None
        else tuned.get("arm_bias_strength", 0.5)
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if arm_bias_strength < 0.0:
        raise ValueError("arm-bias-strength must be non-negative")
    manifest["hyperparameters"] = {
        "gamma": gamma, "arm_bias_strength": arm_bias_strength,
        "source": "cli_override" if args.gamma is not None or args.arm_bias_strength is not None
        else "discovery_summary",
        "discovery_hyperparameters": tuned,
    }

    starts = list(range(lo0, hi_total, args.query_shard_size))
    if args.max_shards is not None:
        starts = starts[:max(0, int(args.max_shards))]
    for lo in starts:
        hi = min(lo + args.query_shard_size, hi_total)
        stem = f"shard_{lo:012d}_{hi:012d}"
        input_path = inputs / f"{stem}.npz"
        score_path = scores / f"{stem}.npz"
        shard_record: dict[str, object] = {"query_start": lo, "query_end": hi,
                                           "input": str(input_path), "scores": str(score_path)}
        # A compact score sidecar is sufficient for a backend rerun.  In
        # particular, merged electricity runs deliberately delete the large
        # history-only input NPZs; do not regenerate them merely to discover
        # that the score already exists.
        if not input_path.exists() and not score_path.exists():
            command = [
                sys.executable, str(ROOT / "scripts/information_anchor/prepare_official_mi_shard.py"),
                "--dataset", args.dataset, "--data-dir", str(Path(args.data_dir).resolve()),
                "--split", args.split,
                "--reference-config", str(Path(args.reference_config).resolve()),
                "--discovery-dir", str(discovery), "--output", str(input_path),
                "--work-dir", str(work_root),
                "--pool-k", str(args.pool_k), "--query-start", str(lo), "--query-end", str(hi),
                "--batch-size", str(args.batch_size), "--seed", str(args.seed),
            ]
            if args.device is not None:
                command += ["--device", args.device]
            shard_record["prepare"] = _run(command, root / f"logs/{stem}_prepare.log")
        elif score_path.exists():
            shard_record["prepare"] = {"status": "reused_from_score"}
        else:
            shard_record["prepare"] = {"status": "reused"}
        if shard_record["prepare"]["status"] not in {"completed", "reused", "reused_from_score"}:
            manifest["shards"].append(shard_record); break
        if not score_path.exists():
            command = [
                sys.executable, str(ROOT / "scripts/information_anchor/build_official_rag_mi_artifact.py"),
                "--input-npz", str(input_path), "--output", str(score_path),
                "--dataset", args.dataset, "--split", args.split,
                "--top-k", str(args.top_k),
                "--seed", str(args.seed), "--mi-target",
                (
                    "I(H;Y|K_recent)" if args.mi_target == "future_truth" and args.mi_condition == "recent"
                    else "I(H;Y)" if args.mi_target == "future_truth"
                    else "I(H;E|K_recent)" if args.mi_condition == "recent"
                    else "I(H;E)"
                ),
                "--mi-condition", f"recent_{args.recent_length}" if args.mi_condition == "recent" else "none",
                "--mi-target-source", args.mi_target,
                "--gamma", str(gamma),
                "--mmr-lambda", str(args.mmr_lambda),
            ]
            shard_record["build"] = _run(command, root / f"logs/{stem}_build.log")
        else:
            shard_record["build"] = {"status": "reused"}
        if (not args.keep_intermediates and
                shard_record["build"]["status"] in {"completed", "reused"}):
            # Score sidecars are the reproducible runner input.  Hidden arrays
            # and raw token NPZs are optional intermediates and otherwise grow
            # to terabytes on the electricity matrix.
            for temporary in (
                work_root / f"query_hidden_{lo:012d}_{hi:012d}",
                work_root / f"candidate_hidden_{lo:012d}_{hi:012d}",
                input_path,
                input_path.with_suffix(".json"),
            ):
                if temporary.is_dir():
                    shutil.rmtree(temporary)
                elif temporary.exists():
                    temporary.unlink()
            shard_record["intermediates_cleaned"] = True
        manifest["shards"].append(shard_record)
    score_status = len(manifest["shards"]) == len(starts) and all(
        item.get("build", {}).get("status") in {"completed", "reused"}
        for item in manifest["shards"]
    )
    manifest["score_directory_ready"] = bool(score_status)
    if args.backend != "none" and score_status:
        data_dir = Path(args.data_dir).resolve()
        if args.backend == "align":
            command = [
                sys.executable, str(ROOT / "scripts/information_anchor/run_official_align_mi.py"),
                "--dataset", args.dataset, "--data-dir", str(data_dir),
                "--split", args.split,
                "--artifact", str(scores), "--results-dir", str(backend_results),
                "--methods", args.methods, "--gpu", str(args.gpu),
                "--pool-k", str(args.pool_k), "--top-k", str(args.top_k),
                "--query-start", str(lo0), "--query-end", str(hi_total),
                "--mmr-lambda", str(args.mmr_lambda),
            ]
            if args.save_preds:
                command += ["--save-preds"]
            if args.mi_blend_pool:
                command += ["--mi-blend-pool"]
            if args.branch_mode is not None:
                command += ["--branch-mode", args.branch_mode]
            if args.blend_score_eta:
                command += ["--blend-score-eta", str(args.blend_score_eta)]
            if args.correction_gain_file:
                command += ["--correction-gain-file", str(Path(args.correction_gain_file).resolve())]
        else:
            command = [
                sys.executable, str(ROOT / "scripts/information_anchor/run_official_tsrag_mi.py"),
                "--dataset", args.dataset, "--root-path", str(data_dir),
                "--data-path", f"{args.dataset}_retrieve.csv", "--data", args.data_kind,
                "--artifact", str(scores), "--retrieval-database-dir", str(Path(args.retrieval_database_dir).resolve()),
                "--metadata-frequency", args.metadata_frequency, "--results-dir", str(backend_results),
                "--methods", args.methods, "--pretrained-model-path", args.pretrained_model_path,
                "--retrieval-checkpoint", str(Path(args.retrieval_checkpoint).resolve()),
                "--split", args.split, "--gpu", str(args.gpu),
                "--pool-k", str(args.pool_k), "--top-k", str(args.top_k),
                "--query-start", str(lo0), "--query-end", str(hi_total),
            ]
            if args.arm_bias:
                command += ["--arm-bias", "--arm-bias-strength", str(arm_bias_strength),
                            "--arm-bias-method", args.arm_bias_method]
            if args.arm_attention_prior:
                command += [
                    "--arm-attention-prior",
                    "--arm-attention-prior-method", args.arm_attention_prior_method,
                    "--arm-attention-prior-strength", str(args.arm_attention_prior_strength),
                ]
            if args.save_preds:
                command += ["--save-preds"]
        manifest["runner"] = _run(command, root / "logs/official_runner.log")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "score_directory_ready": score_status,
                      "shards": len(manifest["shards"]), "runner": manifest.get("runner", {}).get("status", "not_run")}, indent=2))


if __name__ == "__main__":
    main()

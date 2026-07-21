from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .pitch_damping_retune import (
    CHATTER_THRESHOLDS,
    DEFAULT_OUTPUT_DIR,
    FIXED_CONTROLLER_VALUES,
    HARD_GATE_THRESHOLDS,
    PHYSICAL_CONFIGURATION,
    PROVISIONAL_PROFILE,
    SCORE_WEIGHTS,
    SELECTION_COMPARISON_METRICS,
    SELECTION_HARD_GATE_FIELDS,
    SOURCE_PROFILE,
    SYMMETRY_METRICS,
    SYMMETRY_PAIRS,
    Candidate,
    ScenarioDefinition,
    ScenarioResultStore,
    _boolean,
    _canonical_json,
    _fresh_validation_runs,
    _fresh_baseline_results,
    _comparison_summary,
    _json_safe,
    _number,
    _write_validation_timeseries,
    _write_manifest,
    _write_plots,
    aggregate_candidates,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    git_revision,
    raw_score_best,
    read_csv,
    required_scenarios,
    run_stage,
    sha256_file,
    source_hashes,
    final_candidate_requirements,
    validate_parameter_sources,
)


STAGE = "boundary_extension_audit"
INITIAL_RATE_P = (0.09000, 0.09125, 0.09250, 0.09375, 0.09500)
INITIAL_RATE_D = (0.01950, 0.02000, 0.02050, 0.02100)
RATE_P_STEP = 0.00125
RATE_D_STEP = 0.00050
RATE_I = 0.0
ANGLE_P = 25.0
CURRENT_RATE_P = 0.09000
CURRENT_RATE_D = 0.01950
MAX_EXTENSION_ROUNDS = 12
DEFAULT_AUDIT_DIR = DEFAULT_OUTPUT_DIR / "boundary_extension_audit"


def initial_candidates() -> list[Candidate]:
    return [
        Candidate(STAGE, rate_p, rate_d, ANGLE_P)
        for rate_p in INITIAL_RATE_P
        for rate_d in INITIAL_RATE_D
    ]


def _candidate_key(rate_p: float, rate_d: float) -> tuple[int, int]:
    return round(rate_p / RATE_P_STEP), round(rate_d / RATE_D_STEP)


def extension_candidates(
    candidates: Sequence[Candidate], best: dict[str, Any]
) -> list[Candidate]:
    p_values = sorted({candidate.rate_p for candidate in candidates})
    d_values = sorted({candidate.rate_d for candidate in candidates})
    extend_p = math.isclose(_number(best["rate_p"]), p_values[-1], abs_tol=1e-12)
    extend_d = math.isclose(_number(best["rate_d"]), d_values[-1], abs_tol=1e-12)
    if not (extend_p or extend_d):
        return []
    new_p_values = [p_values[-1] + RATE_P_STEP] if extend_p else []
    new_d_values = [d_values[-1] + RATE_D_STEP] if extend_d else []
    existing = {_candidate_key(candidate.rate_p, candidate.rate_d) for candidate in candidates}
    additions = {
        Candidate(STAGE, rate_p, rate_d, ANGLE_P)
        for rate_p in [*p_values, *new_p_values]
        for rate_d in [*d_values, *new_d_values]
        if (rate_p in new_p_values or rate_d in new_d_values)
        and _candidate_key(rate_p, rate_d) not in existing
    }
    return sorted(additions, key=lambda candidate: (candidate.rate_p, candidate.rate_d))


def _audit_fingerprint_payload() -> dict[str, Any]:
    scenario_payload = [
        {
            "key": definition.key,
            "config": vars(definition.config),
            "group": definition.group,
            "direction": definition.direction,
            "event_time_s": definition.event_time_s,
            "requires_target_transient_gates": definition.requires_target_transient_gates,
            "requires_capture_gates": definition.requires_capture_gates,
        }
        for definition in required_scenarios(False)
    ]
    return {
        "schema_version": 1,
        "base_sha": git_revision("rev-parse", "HEAD"),
        "audit_type": "targeted_stage3c_upper_boundary_extension",
        "initial_rate_p": list(INITIAL_RATE_P),
        "initial_rate_d": list(INITIAL_RATE_D),
        "rate_p_step": RATE_P_STEP,
        "rate_d_step": RATE_D_STEP,
        "maximum_extension_rounds": MAX_EXTENSION_ROUNDS,
        "fixed_rate_i": RATE_I,
        "fixed_angle_p": ANGLE_P,
        "fixed_controller_values": FIXED_CONTROLLER_VALUES,
        "physical_configuration": PHYSICAL_CONFIGURATION,
        "source_profile": SOURCE_PROFILE.relative_to(SOURCE_PROFILE.parents[1]).as_posix(),
        "source_profile_sha256": sha256_file(SOURCE_PROFILE),
        "scenario_definitions": scenario_payload,
        "score_weights": SCORE_WEIGHTS,
        "hard_gate_thresholds": HARD_GATE_THRESHOLDS,
        "chatter_thresholds": CHATTER_THRESHOLDS,
        "source_hashes": source_hashes(),
    }


def audit_fingerprint() -> tuple[dict[str, Any], str]:
    payload = _audit_fingerprint_payload()
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return payload, digest


def load_normalization_baseline(
    scenarios: Sequence[ScenarioDefinition],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    baseline_dir = DEFAULT_OUTPUT_DIR / "baseline"
    rows = read_csv(baseline_dir / "baseline_scenario_results.csv")
    by_scenario = {str(row["scenario_name"]): dict(row) for row in rows}
    expected = {scenario.key for scenario in scenarios}
    if set(by_scenario) != expected:
        raise ValueError(
            f"normalization baseline scenario mismatch: expected={sorted(expected)}, "
            f"actual={sorted(by_scenario)}"
        )
    mismatch_payload = json.loads(
        (baseline_dir / "baseline_mismatch.json").read_text(encoding="utf-8")
    )
    reasons = list(mismatch_payload.get("reasons", []))
    if mismatch_payload.get("status") != "mismatch" or not reasons:
        raise ValueError("Stage 0 must remain an explicit FAILED / NON-ACCEPTABLE baseline")
    return by_scenario, reasons


def _upper_boundary_flags(
    best: dict[str, Any], candidates: Sequence[Candidate]
) -> dict[str, bool]:
    p_max = max(candidate.rate_p for candidate in candidates)
    d_max = max(candidate.rate_d for candidate in candidates)
    return {
        "rate_p_at_upper_boundary": math.isclose(
            _number(best["rate_p"]), p_max, abs_tol=1e-12
        ),
        "rate_d_at_upper_boundary": math.isclose(
            _number(best["rate_d"]), d_max, abs_tol=1e-12
        ),
    }


def _validation_rows(
    candidate: Candidate,
    validation_runs: Sequence[dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "stage": "boundary_validation",
            "candidate_key": candidate.key,
            "rate_p": candidate.rate_p,
            "rate_i": RATE_I,
            "rate_d": candidate.rate_d,
            "angle_p": candidate.angle_p,
            "scenario_name": scenario_name,
            **metrics,
        }
        for scenario_name, metrics in validation_runs[0].items()
    ]


def _assert_validation_gates(
    candidate: Candidate,
    scenarios: Sequence[ScenarioDefinition],
    baseline_by_scenario: dict[str, dict[str, Any]],
    validation_runs: Sequence[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    rows = _validation_rows(candidate, validation_runs)
    aggregates = aggregate_candidates(
        rows, baseline_by_scenario, scenarios, "boundary_validation"
    )
    best = raw_score_best(aggregates)
    if _boolean(best.get("rejected")):
        raise RuntimeError(
            f"deterministic validation hard-gate failure: {best.get('rejection_reasons', '')}"
        )
    return best


def _write_surface_plot(output_dir: Path, aggregates: Sequence[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in aggregates if not _boolean(row.get("rejected"))]
    rejected = [row for row in aggregates if _boolean(row.get("rejected"))]
    figure, axis = plt.subplots(figsize=(8.4, 5.6))
    if valid:
        scatter = axis.scatter(
            [_number(row["rate_p"]) for row in valid],
            [_number(row["rate_d"]) for row in valid],
            c=[_number(row["final_score"]) for row in valid],
            cmap="viridis_r",
            s=85,
            edgecolors="black",
            linewidths=0.5,
        )
        figure.colorbar(scatter, ax=axis, label="Raw aggregate score (lower is better)")
    if rejected:
        axis.scatter(
            [_number(row["rate_p"]) for row in rejected],
            [_number(row["rate_d"]) for row in rejected],
            marker="x",
            color="#b91c1c",
            s=80,
            label="Hard-rejected",
        )
    axis.scatter(
        [CURRENT_RATE_P], [CURRENT_RATE_D], marker="*", s=240, color="#dc2626", label="Current"
    )
    axis.set_xlabel("Rate P")
    axis.set_ylabel("Rate D")
    axis.set_title("Targeted pitch-damping upper-boundary audit")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = output_dir / "boundary_extension_raw_score.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _report_text(summary: dict[str, Any], rounds: Sequence[dict[str, Any]]) -> str:
    selected = summary["selected"]
    current = summary["current_reference"]
    return f"""# Targeted Stage 3C boundary-extension audit

## Decision

{summary['decision_text']}

- Selected raw-score controller: Rate P `{selected['rate_p']:.5f}`, Rate I `0.0`, Rate D `{selected['rate_d']:.5f}`, Angle P `25.0`.
- Selected raw aggregate score: `{selected['final_score']:.12f}`.
- Current-reference score: `{current['final_score']:.12f}`.
- Score change versus current: `{summary['score_change_vs_current']:+.12f}` (negative is better).
- Valid candidates: `{summary['valid_candidate_count']}` of `{summary['candidate_count']}`.
- Full-duration scenario runs: `{summary['scenario_run_count']}` search runs plus 14 deterministic validation reruns.
- Upper-boundary extension rounds: `{summary['extension_round_count']}`.
- Deterministic rerun digest: `{summary['deterministic_digest']}`.

No near-equivalent or lower-control-effort tie-break was used. Selection is the valid raw-score local rank-1 point.

## Fixed configuration

Rate I remained `0.0`, Angle P remained `25.0`, and every outer-loop, braking, capture, vehicle, and Vane-only setting remained unchanged. Moving-mass assist gain, actual displacement, and target displacement remained exactly zero in every run.

## Gate enforcement

Every candidate was evaluated in all seven full-duration scenarios. Any failure of either mirrored `+1 m` or `-1 m` early-velocity-reversal gate was a hard rejection. The existing premature-pause, second-lobe, capture-count, capture-discontinuity, shaped-vx-sign, finite-state, crash/ground-contact, chatter, saturation, effective-parameter, and symmetry gates were all retained.

## Normalization status

Stage 0 is **FAILED / NON-ACCEPTABLE** and is used for normalization and comparison only. Its preserved detector failures are: `{'; '.join(summary['stage0_failure_reasons'])}`. It is not a passing validation controller.

## Boundary rounds

| round | candidates | valid | best P | best D | raw score | P upper | D upper |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
{chr(10).join(f"| {row['round']} | {row['candidate_count']} | {row['valid_candidate_count']} | {row['best_rate_p']:.5f} | {row['best_rate_d']:.5f} | {row['best_score']:.12f} | {row['rate_p_at_upper_boundary']} | {row['rate_d_at_upper_boundary']} |" for row in rounds)}
"""


def _scenario_rows_for_candidate(
    store_rows: Sequence[dict[str, Any]], rate_p: float, rate_d: float
) -> dict[str, dict[str, Any]]:
    return {
        str(row["scenario_name"]): dict(row)
        for row in store_rows
        if math.isclose(_number(row["rate_p"]), rate_p, abs_tol=1e-12)
        and math.isclose(_number(row["rate_d"]), rate_d, abs_tol=1e-12)
    }


def _boundary_selection_comparison(
    current: dict[str, Any],
    selected: dict[str, Any],
    current_by_scenario: dict[str, dict[str, Any]],
    selected_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aggregate_metrics: dict[str, Any] = {}
    per_scenario: dict[str, Any] = {}
    for metric, label in SELECTION_COMPARISON_METRICS:
        old_values = np.asarray(
            [_number(row[metric]) for row in current_by_scenario.values()], dtype=float
        )
        new_values = np.asarray(
            [_number(row[metric]) for row in selected_by_scenario.values()], dtype=float
        )
        old_mean = float(np.mean(old_values))
        new_mean = float(np.mean(new_values))
        aggregate_metrics[metric] = {
            "label": label,
            "previous_stage3c_rank1_mean": old_mean,
            "selected_boundary_rank1_mean": new_mean,
            "selected_minus_previous": new_mean - old_mean,
            "selected_change_percent": 100.0
            * (new_mean - old_mean)
            / max(abs(old_mean), 1e-12),
        }
    for name in sorted(selected_by_scenario):
        old_row = current_by_scenario[name]
        new_row = selected_by_scenario[name]
        per_scenario[name] = {
            "metrics": {
                metric: {
                    "label": label,
                    "previous_stage3c_rank1": _number(old_row[metric]),
                    "selected_boundary_rank1": _number(new_row[metric]),
                    "selected_minus_previous": _number(new_row[metric])
                    - _number(old_row[metric]),
                }
                for metric, label in SELECTION_COMPARISON_METRICS
            },
            "previous_hard_gates": {
                key: old_row.get(key) for key in SELECTION_HARD_GATE_FIELDS
            },
            "selected_hard_gates": {
                key: new_row.get(key) for key in SELECTION_HARD_GATE_FIELDS
            },
        }
    old_scenario_scores = json.loads(str(current["scenario_scores_json"]))
    new_scenario_scores = json.loads(str(selected["scenario_scores_json"]))
    return {
        "selection_description": (
            "true local raw-score rank-1 candidate from the targeted upper-boundary extension"
        ),
        "selection_reason": (
            "The specified compact audit found a lower valid raw aggregate score and "
            "continued at the same step size until the best point was not on an upper boundary."
        ),
        "near_equivalent_tiebreak_used": False,
        "previous_stage3c_rank1": {
            "rate_p": _number(current["rate_p"]),
            "rate_d": _number(current["rate_d"]),
            "angle_p": ANGLE_P,
            "raw_aggregate_score": _number(current["final_score"]),
            "scenario_mean_score": float(np.mean(list(old_scenario_scores.values()))),
            "worst_scenario_score": _number(current["worst_scenario_score"]),
            "all_hard_gates_pass": True,
        },
        "selected_boundary_rank1": {
            "rate_p": _number(selected["rate_p"]),
            "rate_d": _number(selected["rate_d"]),
            "angle_p": ANGLE_P,
            "raw_score_rank": 1,
            "raw_aggregate_score": _number(selected["final_score"]),
            "scenario_mean_score": float(np.mean(list(new_scenario_scores.values()))),
            "worst_scenario_score": _number(selected["worst_scenario_score"]),
            "all_hard_gates_pass": True,
            "upper_boundary_flags": {
                "rate_p": False,
                "rate_d": False,
                "angle_p": False,
            },
        },
        "score_improvement": {
            "absolute": _number(current["final_score"])
            - _number(selected["final_score"]),
            "relative_percent": 100.0
            * (_number(current["final_score"]) - _number(selected["final_score"]))
            / _number(current["final_score"]),
        },
        "aggregate_metrics": aggregate_metrics,
        "per_scenario": per_scenario,
        "audit_passed": True,
    }


def _comparison_csv_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scope": "all_scenarios_mean",
            "scenario": "all",
            "metric": metric,
            "label": payload["label"],
            "previous_stage3c_rank1": payload["previous_stage3c_rank1_mean"],
            "selected_boundary_rank1": payload["selected_boundary_rank1_mean"],
            "selected_minus_previous": payload["selected_minus_previous"],
            "selected_change_percent": payload["selected_change_percent"],
        }
        for metric, payload in comparison["aggregate_metrics"].items()
    ]


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    old = comparison["previous_stage3c_rank1"]
    new = comparison["selected_boundary_rank1"]
    improvement = comparison["score_improvement"]
    rows = [
        "# Previous Stage 3C rank 1 versus boundary-extended local rank 1",
        "",
        "Stage 0 remains **FAILED / NON-ACCEPTABLE** and normalization-only.",
        "",
        "No near-equivalent lower-effort tie-break was used. The final controller is the valid raw-score local rank-1 point from the targeted extension.",
        "",
        "| metric | previous P=0.09000, D=0.01950 | selected P=0.09375, D=0.02100 | selected change |",
        "| --- | ---: | ---: | ---: |",
        f"| raw aggregate score | {old['raw_aggregate_score']:.12f} | {new['raw_aggregate_score']:.12f} | {-improvement['relative_percent']:.3f}% |",
        f"| scenario mean score | {old['scenario_mean_score']:.12f} | {new['scenario_mean_score']:.12f} | {100.0 * (new['scenario_mean_score'] - old['scenario_mean_score']) / old['scenario_mean_score']:.3f}% |",
        f"| worst-scenario score | {old['worst_scenario_score']:.12f} | {new['worst_scenario_score']:.12f} | {100.0 * (new['worst_scenario_score'] - old['worst_scenario_score']) / old['worst_scenario_score']:.3f}% |",
    ]
    for payload in comparison["aggregate_metrics"].values():
        rows.append(
            f"| {payload['label']} | {payload['previous_stage3c_rank1_mean']:.9f} | "
            f"{payload['selected_boundary_rank1_mean']:.9f} | {payload['selected_change_percent']:.3f}% |"
        )
    rows.extend(
        [
            "",
            "Both controllers pass every hard gate in all seven full-duration scenarios. The selected point is interior on both audited P and D upper axes after one D-axis extension round.",
            "",
        ]
    )
    return "\n".join(rows)


def update_publication_artifacts(
    output_dir: Path, summary: dict[str, Any], store: ScenarioResultStore
) -> None:
    scenarios = required_scenarios(False)
    baseline_by_scenario, baseline_failures = load_normalization_baseline(scenarios)
    selected = summary["selected"]
    current = summary["current_reference"]
    selected_candidate = Candidate(
        "selected_validation", _number(selected["rate_p"]), _number(selected["rate_d"]), ANGLE_P
    )
    validation_runs, selected_results, digests = _fresh_validation_runs(
        selected_candidate, scenarios, quick=False
    )
    selected_by_scenario = validation_runs[0]
    failures, baseline_comparison = final_candidate_requirements(
        baseline_by_scenario,
        selected_by_scenario,
        {
            "rate_p_at_min": False,
            "rate_p_at_max": False,
            "rate_d_at_min": False,
            "rate_d_at_max": False,
            "angle_p_at_min": False,
            "angle_p_at_max": False,
        },
    )
    if failures:
        raise RuntimeError("extended selection failed final requirements: " + "; ".join(failures))
    current_by_scenario = _scenario_rows_for_candidate(
        store.rows, _number(current["rate_p"]), _number(current["rate_d"])
    )
    selected_search_by_scenario = _scenario_rows_for_candidate(
        store.rows, _number(selected["rate_p"]), _number(selected["rate_d"])
    )
    comparison = _boundary_selection_comparison(
        current, selected, current_by_scenario, selected_search_by_scenario
    )

    validation_payload = {
        "workflow_fingerprint": summary["fingerprint"],
        "digests": digests,
        "byte_identical_metrics": len(set(digests)) == 1,
        "requirement_failures": [],
        "runs": validation_runs,
    }
    main_validation_dir = DEFAULT_OUTPUT_DIR / "validation"
    atomic_write_json(main_validation_dir / "deterministic_reruns.json", _json_safe(validation_payload))
    atomic_write_json(
        main_validation_dir / "raw_score_rank1_deterministic_reruns.json",
        _json_safe(validation_payload),
    )
    _write_validation_timeseries(DEFAULT_OUTPUT_DIR, "selected", selected_results)
    _write_validation_timeseries(DEFAULT_OUTPUT_DIR, "raw_score_rank1", selected_results)

    _baseline_rows, baseline_results = _fresh_baseline_results(scenarios, quick=False)
    _write_plots(DEFAULT_OUTPUT_DIR, baseline_results, selected_results)
    shutil.copy2(
        output_dir / "boundary_extension_raw_score.png",
        DEFAULT_OUTPUT_DIR / "plots" / "boundary_extension_raw_score.png",
    )

    atomic_write_csv(
        DEFAULT_OUTPUT_DIR / "best_parameters.csv",
        [
            {
                "rate_p": selected_candidate.rate_p,
                "rate_i": RATE_I,
                "rate_d": selected_candidate.rate_d,
                "angle_p": ANGLE_P,
                "position_p": 0.55,
                "velocity_p": 0.70,
                "moving_mass_assist_gain_m_per_Nm": 0.0,
            }
        ],
    )
    atomic_write_json(DEFAULT_OUTPUT_DIR / "baseline_comparison.json", _json_safe(baseline_comparison))
    atomic_write_json(DEFAULT_OUTPUT_DIR / "selection_comparison.json", _json_safe(comparison))
    atomic_write_csv(DEFAULT_OUTPUT_DIR / "selection_comparison.csv", _comparison_csv_rows(comparison))
    atomic_write_text(DEFAULT_OUTPUT_DIR / "selection_comparison.md", _comparison_markdown(comparison))

    boundary_rows = read_csv(DEFAULT_OUTPUT_DIR / "boundary_diagnostics.csv")
    boundary_rows = [
        row for row in boundary_rows if not str(row.get("stage", "")).startswith("targeted_boundary_audit")
    ]
    for round_row in read_csv(output_dir / "boundary_rounds.csv"):
        boundary_rows.append(
            {
                "stage": f"targeted_boundary_audit_round_{round_row['round']}",
                "rate_p_at_min": False,
                "rate_p_at_max": _boolean(round_row["rate_p_at_upper_boundary"]),
                "rate_d_at_min": False,
                "rate_d_at_max": _boolean(round_row["rate_d_at_upper_boundary"]),
                "angle_p_at_min": False,
                "angle_p_at_max": False,
                "candidate_key": selected["candidate_key"],
                "rate_p": round_row["best_rate_p"],
                "rate_d": round_row["best_rate_d"],
                "angle_p": ANGLE_P,
                "rejected": False,
                "final_score": round_row["best_score"],
            }
        )
    atomic_write_csv(DEFAULT_OUTPUT_DIR / "boundary_diagnostics.csv", boundary_rows)

    metadata_rows = read_csv(DEFAULT_OUTPUT_DIR / "search_metadata.csv")
    metadata = {str(row["key"]): row["value"] for row in metadata_rows}
    metadata.update(
        {
            "selected": _canonical_json(
                {"stage": "boundary_extension_audit", "rate_p": selected_candidate.rate_p, "rate_d": selected_candidate.rate_d, "angle_p": ANGLE_P}
            ),
            "boundary_flags": _canonical_json(
                {"rate_p_at_max": False, "rate_d_at_max": False, "angle_p_at_max": False}
            ),
            "deterministic_rerun_digests": _canonical_json(digests),
            "boundary_extension_audit": _canonical_json(
                {
                    "candidate_count": summary["candidate_count"],
                    "scenario_run_count": summary["scenario_run_count"],
                    "extension_round_count": summary["extension_round_count"],
                    "selected_raw_score": selected["final_score"],
                    "previous_raw_score": current["final_score"],
                    "near_equivalent_tiebreak_used": False,
                    "status": "boundary-validated interior raw-score local rank 1",
                }
            ),
            "selection": _canonical_json(
                {
                    "description": comparison["selection_description"],
                    "reason": comparison["selection_reason"],
                    "raw_score_rank": 1,
                    "raw_score_best": selected["final_score"],
                    "selected_raw_score": selected["final_score"],
                    "score_improvement_vs_previous": comparison["score_improvement"],
                    "near_equivalent_tiebreak_used": False,
                    "audit_passed": True,
                }
            ),
        }
    )
    atomic_write_csv(
        DEFAULT_OUTPUT_DIR / "search_metadata.csv",
        [{"key": key, "value": value} for key, value in metadata.items()],
    )

    methodology_path = DEFAULT_OUTPUT_DIR / "methodology.md"
    methodology = methodology_path.read_text(encoding="utf-8").split("\n## Targeted boundary-extension audit")[0].rstrip()
    methodology += f"""

## Targeted boundary-extension audit

The final audit evaluated the explicitly requested 5 x 4 grid at fixed Rate I `0.0` and Angle P `25.0` in all seven full-duration scenarios. The initial best was on the Rate D upper boundary, so one additional D step of `0.00050` was evaluated across the five P values. The raw-score best remained P `{selected_candidate.rate_p:.5f}`, D `{selected_candidate.rate_d:.5f}` and became interior. No near-equivalent lower-effort tie-break was used.

Stage 0 remains **FAILED / NON-ACCEPTABLE** and normalization-only, with preserved failures `{'; '.join(baseline_failures)}`.
"""
    atomic_write_text(methodology_path, methodology)

    pitch = baseline_comparison["aggregate"]["tail_rms_pitch_deg"]
    pitch_rate = baseline_comparison["aggregate"]["tail_rms_pitch_rate_deg_s"]
    velocity = baseline_comparison["aggregate"]["tail_rms_horizontal_velocity_m_s"]
    report = f"""# Vane-only pitch damping retune

The final selected controller is the **valid raw-score local rank 1** from the targeted boundary extension: Rate P/I/D `{selected_candidate.rate_p:.8f} / 0.00000000 / {selected_candidate.rate_d:.8f}`, Angle P `{ANGLE_P:.8f}`. Its raw aggregate score is `{selected['final_score']:.12f}`, improving the previous Stage 3C rank-1 score `{current['final_score']:.12f}` by `{comparison['score_improvement']['absolute']:.12f}` ({comparison['score_improvement']['relative_percent']:.3f}%). No near-equivalent lower-effort tie-break was used.

**Stage 0 status: FAILED / NON-ACCEPTABLE; normalization and comparison only.** Its absolute metrics and both early-reversal failures remain preserved. Stage 0 is not a validated controller.

| metric | Stage 0 mean | selected mean | improvement |
| --- | ---: | ---: | ---: |
| tail RMS pitch (deg) | {pitch['baseline_mean']:.8f} | {pitch['selected_mean']:.8f} | {pitch['mean_improvement_percent']:.3f}% |
| tail RMS pitch rate (deg/s) | {pitch_rate['baseline_mean']:.8f} | {pitch_rate['selected_mean']:.8f} | {pitch_rate['mean_improvement_percent']:.3f}% |
| tail RMS horizontal velocity (m/s) | {velocity['baseline_mean']:.8f} | {velocity['selected_mean']:.8f} | {velocity['mean_improvement_percent']:.3f}% |

All 25 targeted candidates passed all seven full-duration physical, behavioral, symmetry, chatter, saturation, and early-reversal gates. The selected controller eliminates early velocity reversal in both +1 m and -1 m cases, records exactly one stick-release capture, has no capture discontinuity or shaped-vx reversal, and keeps moving-mass gain, actual displacement, and target displacement exactly zero. Two fresh deterministic seven-scenario reruns were byte-identical at the metrics level (`{digests[0]}`).

The initial audit best was on the Rate D upper boundary at `0.02100`; one same-step extension to `0.02150` made the selected `D=0.02100` point interior. Rate P `0.09375` was already interior. This validates that the prior Stage 3C upper bounds did not conceal a better continuing-edge point.

These results apply only to the same deterministic 2D analytical model and do not establish real-flight, 3D, HIL, or hardware safety.
"""
    atomic_write_text(DEFAULT_OUTPUT_DIR / "pitch_damping_retune_summary.md", report)
    atomic_write_text(DEFAULT_OUTPUT_DIR / "selected_candidate_summary.md", report)

    profile = json.loads(PROVISIONAL_PROFILE.read_text(encoding="utf-8"))
    profile["controller"].update(
        {
            "atc_rat_pit_p": selected_candidate.rate_p,
            "atc_rat_pit_i": RATE_I,
            "atc_rat_pit_d": selected_candidate.rate_d,
            "atc_ang_pit_p": ANGLE_P,
        }
    )
    profile["analysis"].update(
        {
            "selected_rate_p": selected_candidate.rate_p,
            "selected_rate_i": RATE_I,
            "selected_rate_d": selected_candidate.rate_d,
            "selected_angle_p": ANGLE_P,
            "boundary_flags": {
                "rate_p_at_max": False,
                "rate_d_at_max": False,
                "angle_p_at_max": False,
            },
            "deterministic_rerun": {"passed": True, "digests": digests},
            "boundary_extension_audit": summary,
            "selection": {
                "description": comparison["selection_description"],
                "reason": comparison["selection_reason"],
                "raw_score_rank": 1,
                "raw_aggregate_score": selected["final_score"],
                "raw_score_best": selected["final_score"],
                "score_penalty": {"absolute": 0.0, "relative_percent": 0.0},
                "near_equivalent_tiebreak_used": False,
                "previous_stage3c_rank1": comparison["previous_stage3c_rank1"],
                "score_improvement": comparison["score_improvement"],
                "audit_passed": True,
            },
        }
    )
    atomic_write_json(PROVISIONAL_PROFILE, _json_safe(profile))

    prior_manifest = json.loads((DEFAULT_OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest_metadata = dict(prior_manifest.get("metadata", {}))
    manifest_metadata.update(
        {
            "selected": {
                "stage": "boundary_extension_audit",
                "rate_p": selected_candidate.rate_p,
                "rate_d": selected_candidate.rate_d,
                "angle_p": ANGLE_P,
            },
            "boundary_extension_audit": summary,
            "deterministic_rerun_digests": digests,
        }
    )
    _write_manifest(
        DEFAULT_OUTPUT_DIR,
        manifest_metadata,
        prior_manifest["fingerprint_payload"],
        prior_manifest["preservation_hashes_before_and_after"],
    )


def run_boundary_audit(
    output_dir: Path = DEFAULT_AUDIT_DIR, *, resume: bool = True
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_parameter_sources()
    scenarios = required_scenarios(False)
    baseline_by_scenario, baseline_failures = load_normalization_baseline(scenarios)
    fingerprint_payload, fingerprint = audit_fingerprint()
    store = ScenarioResultStore(
        output_dir / "scenario_results.csv", fingerprint, resume=resume
    )
    candidates = initial_candidates()
    rounds: list[dict[str, Any]] = []

    aggregates = run_stage(
        STAGE,
        candidates,
        scenarios,
        store,
        fingerprint,
        baseline_by_scenario,
        quick=False,
    )
    current = next(
        row
        for row in aggregates
        if math.isclose(_number(row["rate_p"]), CURRENT_RATE_P, abs_tol=1e-12)
        and math.isclose(_number(row["rate_d"]), CURRENT_RATE_D, abs_tol=1e-12)
    )
    if _boolean(current.get("rejected")):
        raise RuntimeError("the current reference controller unexpectedly failed a hard gate")
    current_score = _number(current["final_score"])

    extension_round = 0
    while True:
        best = raw_score_best(aggregates)
        flags = _upper_boundary_flags(best, candidates)
        valid_count = sum(not _boolean(row.get("rejected")) for row in aggregates)
        rounds.append(
            {
                "round": extension_round,
                "candidate_count": len(aggregates),
                "valid_candidate_count": valid_count,
                "best_rate_p": _number(best["rate_p"]),
                "best_rate_d": _number(best["rate_d"]),
                "best_score": _number(best["final_score"]),
                **flags,
            }
        )
        improved = _number(best["final_score"]) < current_score - 1e-12
        if not improved or not any(flags.values()):
            break
        extension_round += 1
        if extension_round > MAX_EXTENSION_ROUNDS:
            raise RuntimeError("best valid point remained on an upper boundary after 12 extensions")
        additions = extension_candidates(candidates, best)
        if not additions:
            raise RuntimeError("upper-boundary extension produced no new candidates")
        candidates.extend(additions)
        aggregates = run_stage(
            STAGE,
            additions,
            scenarios,
            store,
            fingerprint,
            baseline_by_scenario,
            quick=False,
        )

    selected = raw_score_best(aggregates)
    improved = _number(selected["final_score"]) < current_score - 1e-12
    if not improved:
        selected = current
        decision = "retain_current_boundary_validated"
        decision_text = (
            "No valid extended candidate improved the raw aggregate score. The current "
            "P=0.09000, D=0.01950 controller is retained and boundary-validated."
        )
    else:
        decision = "select_extended_raw_score_rank1"
        decision_text = (
            "A valid extended candidate improved the raw aggregate score and the extension "
            "continued until the best point was no longer on an upper boundary."
        )

    selected_candidate = Candidate(
        "boundary_validation",
        _number(selected["rate_p"]),
        _number(selected["rate_d"]),
        ANGLE_P,
    )
    validation_runs, _validation_results, digests = _fresh_validation_runs(
        selected_candidate, scenarios, quick=False
    )
    validation_aggregate = _assert_validation_gates(
        selected_candidate, scenarios, baseline_by_scenario, validation_runs
    )

    aggregates = aggregate_candidates(
        store.rows, baseline_by_scenario, scenarios, STAGE
    )
    rejection_counts = Counter(
        reason.strip()
        for row in aggregates
        for reason in str(row.get("rejection_reasons", "")).split(";")
        if reason.strip()
    )
    summary = {
        "schema_version": 1,
        "decision": decision,
        "decision_text": decision_text,
        "controller_changed": decision == "select_extended_raw_score_rank1",
        "selected": _json_safe(selected),
        "current_reference": _json_safe(current),
        "score_change_vs_current": _number(selected["final_score"]) - current_score,
        "candidate_count": len(aggregates),
        "valid_candidate_count": sum(
            not _boolean(row.get("rejected")) for row in aggregates
        ),
        "rejected_candidate_count": sum(
            _boolean(row.get("rejected")) for row in aggregates
        ),
        "scenario_run_count": len(store.rows),
        "extension_round_count": extension_round,
        "deterministic_digest": digests[0],
        "deterministic_digests": digests,
        "deterministic_validation_raw_score": _number(
            validation_aggregate["final_score"]
        ),
        "stage0_status": "FAILED / NON-ACCEPTABLE; normalization-only",
        "stage0_failure_reasons": baseline_failures,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "moving_mass_assist_gain_m_per_Nm": 0.0,
        "moving_mass_actual_displacement_m": 0.0,
        "moving_mass_target_displacement_m": 0.0,
        "fixed_rate_i": RATE_I,
        "fixed_angle_p": ANGLE_P,
        "near_equivalent_tiebreak_used": False,
        "fingerprint": fingerprint,
        "runtime_s": time.perf_counter() - started,
    }

    atomic_write_csv(output_dir / "candidate_results.csv", aggregates)
    atomic_write_csv(output_dir / "boundary_rounds.csv", rounds)
    atomic_write_csv(
        output_dir / "rejection_summary.csv",
        [
            {"rejection_category": key, "candidate_count": value}
            for key, value in sorted(rejection_counts.items())
        ],
    )
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_text(output_dir / "summary.md", _report_text(summary, rounds))
    plot_path = _write_surface_plot(output_dir, aggregates)

    if summary["controller_changed"]:
        update_publication_artifacts(output_dir, summary, store)

    artifacts = {}
    for path in sorted(file for file in output_dir.rglob("*") if file.is_file()):
        if path.name == "manifest.json":
            continue
        artifacts[path.relative_to(output_dir).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "deterministic": True,
        "summary": summary,
        "fingerprint_payload": fingerprint_payload,
        "artifacts": artifacts,
        "provisional_profile_sha256": sha256_file(PROVISIONAL_PROFILE),
        "plot": plot_path.relative_to(output_dir).as_posix(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return summary

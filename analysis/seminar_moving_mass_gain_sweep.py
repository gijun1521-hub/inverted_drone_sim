from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .headless_loiter import LoiterScenarioConfig
from .seminar_video_scenarios import (
    ASSIST_GAIN_M_PER_NM,
    SeminarScenarioDefinition,
    SeminarVariantDefinition,
    effective_parameter_mismatches,
    run_seminar_variant,
)


COARSE_GAINS = (0.000, 0.025, 0.040, 0.055, 0.070, 0.085, 0.100)
EXPANSION_GAINS = tuple(round(0.105 + 0.005 * index, 4) for index in range(20))
REFINEMENT_HALF_WIDTH = 0.010
REFINEMENT_STEP = 0.0025
NEAR_EQUIVALENT_SCORE_FRACTION = 0.01

# These engineering scales are fixed before selection. Each normalized metric
# contributes its weight; the aggregate adds half the worst scenario score to
# the four-scenario mean so a one-direction win cannot dominate the choice.
SCORE_TERMS = {
    "tail_rms_x_m": (0.10, 2.00),
    "tail_rms_vx_m_s": (0.20, 1.50),
    "tail_peak_to_peak_x_m": (0.25, 1.00),
    "tail_path_length_m": (0.40, 1.00),
    "final_abs_x_error_m": (0.10, 1.50),
    "position_overshoot_m": (0.50, 0.50),
    "peak_abs_theta_deg": (10.0, 0.75),
    "tail_rms_theta_deg": (4.0, 0.75),
    "vane_command_rms_deg": (0.80, 1.00),
    "vane_command_max_deg": (4.0, 0.25),
    "moving_mass_max_offset_m": (0.05, 0.30),
    "moving_mass_tracking_rms_m": (0.005, 0.20),
}
UNSETTLED_PENALTY = 0.25

CSV_COLUMNS = [
    "gain_m_per_Nm",
    "stages",
    "scenario",
    "direction",
    *SCORE_TERMS,
    "settling_time_s",
    "settled",
    "premature_pause",
    "second_acceleration_lobe_after_full_pause",
    "early_velocity_reversal",
    "moving_mass_rail_saturation",
    "vane_saturation_percent",
    "target_capture_discontinuity",
    "ground_contact",
    "rejected",
    "rejection_reasons",
    "scenario_score",
    "aggregate_score",
    "accepted_all_scenarios",
    "selected",
]


@dataclass(frozen=True)
class GainSweepResult:
    rows: list[dict[str, Any]]
    selected_gain_m_per_Nm: float
    tested_gains_m_per_Nm: tuple[float, ...]
    aggregate_rows: list[dict[str, Any]]

    def metadata(self) -> dict[str, Any]:
        baseline = next(row for row in self.aggregate_rows if math.isclose(row["gain_m_per_Nm"], 0.055))
        return {
            "selected_gain_m_per_Nm": self.selected_gain_m_per_Nm,
            "tested_gains_m_per_Nm": list(self.tested_gains_m_per_Nm),
            "selection_method": "minimum accepted mean scenario score plus 0.5 times worst scenario score; within 1%, prefer lower gain then lower maximum mass travel",
            "score_terms": {
                key: {"scale": scale, "weight": weight}
                for key, (scale, weight) in SCORE_TERMS.items()
            },
            "unsettled_penalty": UNSETTLED_PENALTY,
            "gain_0p055_acceptable": bool(baseline["accepted_all_scenarios"]),
            "gain_0p055_rejection_reasons": baseline["rejection_reasons"],
            "deterministic_fingerprint_sha256": sweep_fingerprint(self.rows),
        }


def validation_scenarios(duration_s: float = 8.0) -> tuple[SeminarScenarioDefinition, ...]:
    def disturbance(key: str, force: float) -> SeminarScenarioDefinition:
        direction = -1 if force > 0 else 1
        return SeminarScenarioDefinition(
            key=key,
            display_name=f"{force:+.0f} N disturbance recovery",
            config=LoiterScenarioConfig(
                name=f"seminar_gain_sweep_{key}",
                duration_s=duration_s,
                initial_x=0.0,
                initial_z=1.0,
                initial_theta_deg=0.0,
                target_x=0.0,
                target_z=1.0,
                disturbance_start_s=1.5,
                disturbance_duration_s=0.2,
                disturbance_force_x=force,
                notes="Mirrored deterministic gain-selection disturbance.",
            ),
            settling_reference_time_s=1.7,
            response_kind="disturbance",
            motion_direction=direction,
        )

    def target_step(key: str, target: float) -> SeminarScenarioDefinition:
        direction = 1 if target > 0 else -1
        return SeminarScenarioDefinition(
            key=key,
            display_name=f"{target:+.0f} m absolute target",
            config=LoiterScenarioConfig(
                name=f"seminar_gain_sweep_{key}",
                duration_s=duration_s,
                initial_x=0.0,
                initial_z=1.0,
                initial_theta_deg=0.0,
                target_x=0.0,
                target_z=1.0,
                target_step_time_s=1.0,
                target_step_x=target,
                notes="Mirrored deterministic absolute target; no external smoothing.",
            ),
            settling_reference_time_s=1.0,
            response_kind="target_step",
            motion_direction=direction,
        )

    return (
        disturbance("disturbance_pos8n", 8.0),
        disturbance("disturbance_neg8n", -8.0),
        target_step("target_pos1m", 1.0),
        target_step("target_neg1m", -1.0),
    )


def scenario_score(metrics: dict[str, Any]) -> float:
    score = sum(
        weight * float(metrics[key]) / scale
        for key, (scale, weight) in SCORE_TERMS.items()
    )
    if not metrics["settled"]:
        score += UNSETTLED_PENALTY
    return float(score)


def _canonical_gain(value: float) -> float:
    return round(float(value) + 0.0, 4)


def _run_gain(
    gain: float,
    scenarios: Iterable[SeminarScenarioDefinition],
) -> list[dict[str, Any]]:
    variant = SeminarVariantDefinition("assist", "Moving-mass assist", "Active moving mass", gain)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = run_seminar_variant(scenario, variant)
        metric = dict(result.metrics)
        mismatches = effective_parameter_mismatches(result)
        if mismatches:
            mismatch_reason = "controller_or_parameter_mismatch:" + ",".join(mismatches)
            metric["rejected"] = True
            metric["rejection_reasons"] = "; ".join(
                part for part in (metric["rejection_reasons"], mismatch_reason) if part
            )
        metric.update(
            {
                "gain_m_per_Nm": gain,
                "scenario": scenario.key,
                "direction": scenario.motion_direction,
                "scenario_score": scenario_score(metric),
            }
        )
        rows.append(metric)
    return rows


def aggregate_gain_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gain: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_gain.setdefault(float(row["gain_m_per_Nm"]), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for gain in sorted(by_gain):
        group = by_gain[gain]
        accepted = len(group) == 4 and not any(bool(row["rejected"]) for row in group)
        scores = [float(row["scenario_score"]) for row in group]
        reasons = sorted(
            {
                reason.strip()
                for row in group
                for reason in str(row["rejection_reasons"]).split(";")
                if reason.strip()
            }
        )
        aggregates.append(
            {
                "gain_m_per_Nm": gain,
                "aggregate_score": float(sum(scores) / len(scores) + 0.5 * max(scores)),
                "accepted_all_scenarios": accepted,
                "max_mass_travel_m": max(float(row["moving_mass_max_offset_m"]) for row in group),
                "rejection_reasons": "; ".join(reasons),
            }
        )
    return aggregates


def select_gain(aggregate_rows: Iterable[dict[str, Any]]) -> float:
    accepted = [row for row in aggregate_rows if bool(row["accepted_all_scenarios"])]
    if not accepted:
        raise RuntimeError("no moving-mass gain passed all four validation scenarios")
    best_score = min(float(row["aggregate_score"]) for row in accepted)
    near = [
        row
        for row in accepted
        if float(row["aggregate_score"]) <= best_score * (1.0 + NEAR_EQUIVALENT_SCORE_FRACTION)
    ]
    chosen = min(
        near,
        key=lambda row: (
            float(row["gain_m_per_Nm"]),
            float(row["max_mass_travel_m"]),
            float(row["aggregate_score"]),
        ),
    )
    return float(chosen["gain_m_per_Nm"])


def run_staged_gain_sweep(duration_s: float = 8.0) -> GainSweepResult:
    scenarios = validation_scenarios(duration_s)
    rows_by_gain: dict[float, list[dict[str, Any]]] = {}
    stages_by_gain: dict[float, set[str]] = {}

    def evaluate(gains: Iterable[float], stage: str) -> None:
        for raw_gain in gains:
            gain = _canonical_gain(raw_gain)
            stages_by_gain.setdefault(gain, set()).add(stage)
            if gain not in rows_by_gain:
                rows_by_gain[gain] = _run_gain(gain, scenarios)

    evaluate(COARSE_GAINS, "coarse")
    aggregates = aggregate_gain_rows(row for group in rows_by_gain.values() for row in group)
    if not any(row["accepted_all_scenarios"] for row in aggregates):
        evaluate(EXPANSION_GAINS, "expanded")

    aggregates = aggregate_gain_rows(row for group in rows_by_gain.values() for row in group)
    expansion_best = select_gain(aggregates)
    refine_count = int(round(2.0 * REFINEMENT_HALF_WIDTH / REFINEMENT_STEP))
    refinement = (
        _canonical_gain(expansion_best - REFINEMENT_HALF_WIDTH + index * REFINEMENT_STEP)
        for index in range(refine_count + 1)
    )
    evaluate(refinement, "refined")

    all_rows = [row for gain in sorted(rows_by_gain) for row in rows_by_gain[gain]]
    aggregates = aggregate_gain_rows(all_rows)
    selected = select_gain(aggregates)
    aggregate_by_gain = {float(row["gain_m_per_Nm"]): row for row in aggregates}
    for row in all_rows:
        gain = float(row["gain_m_per_Nm"])
        aggregate = aggregate_by_gain[gain]
        row["stages"] = "+".join(sorted(stages_by_gain[gain]))
        row["aggregate_score"] = aggregate["aggregate_score"]
        row["accepted_all_scenarios"] = aggregate["accepted_all_scenarios"]
        row["selected"] = math.isclose(gain, selected, abs_tol=1e-12)
    return GainSweepResult(
        rows=all_rows,
        selected_gain_m_per_Nm=selected,
        tested_gains_m_per_Nm=tuple(sorted(rows_by_gain)),
        aggregate_rows=aggregates,
    )


def sweep_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {key: row.get(key) for key in CSV_COLUMNS if key not in {"stages", "selected"}}
        for row in rows
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_gain_sweep_csv(result: GainSweepResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.rows)
    return path


def write_gain_selection_markdown(result: GainSweepResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_gain = {float(row["gain_m_per_Nm"]): row for row in result.aggregate_rows}
    baseline = by_gain[0.055]
    lines = [
        "# Moving-mass assist gain selection",
        "",
        "All candidates use the PR #19 controller and the same 2.0 kg vehicle. The four deterministic validation scenarios are +8 N and -8 N disturbance recovery plus +1 m and -1 m absolute target steps. Mirrored cases are selection-only and are not rendered.",
        "",
        "The staged search starts with 0.000, 0.025, 0.040, 0.055, 0.070, 0.085, and 0.100 m/Nm. Because none passed every hard gate, it expands by 0.005 m/Nm and then refines around the best accepted expanded candidate by 0.0025 m/Nm. The score is the mean normalized scenario score plus 0.5 times the worst scenario score. Strictly unsettled runs receive a 0.25 penalty. Candidates within 1% of the best score prefer lower gain, then lower mass travel.",
        "",
        "## Aggregate results",
        "",
        "| Gain (m/Nm) | Stages | Accepted | Aggregate score | Max mass travel (mm) | Rejection reasons |",
        "|---:|---|:---:|---:|---:|---|",
    ]
    stages = {
        float(row["gain_m_per_Nm"]): row["stages"]
        for row in result.rows[::4]
    }
    for row in result.aggregate_rows:
        gain = float(row["gain_m_per_Nm"])
        lines.append(
            f"| {gain:.4f} | {stages[gain]} | {'yes' if row['accepted_all_scenarios'] else 'no'} | "
            f"{float(row['aggregate_score']):.6f} | {1000.0 * float(row['max_mass_travel_m']):.3f} | "
            f"{row['rejection_reasons'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Per-scenario metrics",
            "",
            "| Gain | Scenario | Tail RMS x | Tail RMS vx | Tail p-p x | Tail path | Final error | Overshoot/excursion | Peak pitch | Tail RMS pitch | Vane RMS | Vane max | Mass max (mm) | Tracking RMS (mm) | Settled | Rejected | Reasons |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
        ]
    )
    for row in result.rows:
        lines.append(
            f"| {float(row['gain_m_per_Nm']):.4f} | {row['scenario']} | "
            f"{float(row['tail_rms_x_m']):.5f} | {float(row['tail_rms_vx_m_s']):.5f} | "
            f"{float(row['tail_peak_to_peak_x_m']):.5f} | {float(row['tail_path_length_m']):.5f} | "
            f"{float(row['final_abs_x_error_m']):.5f} | {float(row['position_overshoot_m']):.5f} | "
            f"{float(row['peak_abs_theta_deg']):.3f} | {float(row['tail_rms_theta_deg']):.3f} | "
            f"{float(row['vane_command_rms_deg']):.3f} | {float(row['vane_command_max_deg']):.3f} | "
            f"{1000.0 * float(row['moving_mass_max_offset_m']):.3f} | "
            f"{1000.0 * float(row['moving_mass_tracking_rms_m']):.3f} | "
            f"{'yes' if row['settled'] else 'no'} | {'yes' if row['rejected'] else 'no'} | "
            f"{row['rejection_reasons'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Selected gain: **{result.selected_gain_m_per_Nm:.4f} m/Nm**. It passes every hard gate in all four directions and minimizes the robust aggregate under the stated near-equivalence preference.",
            "",
            f"The former 0.055 m/Nm gain is **{'acceptable' if baseline['accepted_all_scenarios'] else 'not acceptable'}** in this broader sweep. Rejection reasons: {baseline['rejection_reasons'] or 'none'}.",
            "",
            "Hard gates reject non-finite data, crash or ground contact, the defined premature pause, a second acceleration lobe after a full pause, early velocity reversal, rail saturation, more than 5% vane saturation, capture target discontinuity, or an effective controller/parameter mismatch.",
            "",
            "## Remaining limitations",
            "",
            "This is a deterministic 2D model study, not hardware validation. It does not cover 3D coupling, calibration uncertainty, actuator wear, sensor noise outside the configured model, turbulence, structural motion, or the complete flight envelope. Strict settling is reported without extending the 8-second observation window.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def assert_selected_gain(result: GainSweepResult) -> None:
    if not math.isclose(result.selected_gain_m_per_Nm, ASSIST_GAIN_M_PER_NM, abs_tol=1e-12):
        raise RuntimeError(
            f"gain sweep selected {result.selected_gain_m_per_Nm:.4f}, "
            f"but seminar scenarios declare {ASSIST_GAIN_M_PER_NM:.4f}"
        )

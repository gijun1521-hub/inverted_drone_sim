"""Event-aware final LOITER stick-motion regression for the merged controller."""
from __future__ import annotations

import csv, hashlib, json, math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .headless_loiter import LoiterScenarioConfig, run_headless_loiter, save_loiter_timeseries
from .moving_mass_gain_resweep import FIXED_CONTROLLER, MOVING_MASS_LIMITER_HARD_GATES, _moving_mass_metrics
from .pitch_damping_retune import CHATTER_THRESHOLDS, HARD_GATE_THRESHOLDS, ScenarioDefinition, _canonical_json, _sha256_bytes, compute_metrics
from .loiter_transient_diagnosis import detect_premature_pause, second_acceleration_lobe_after_full_pause

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "params" / "moving_mass_gain_resweep_provisional.json"
OUT = ROOT / "results" / "analysis" / "final_loiter_regression"
VARIANTS = (("vane_only", 0.0), ("moving_mass_assist", 0.0415))

@dataclass(frozen=True)
class Pattern:
    key: str; timeline: tuple[tuple[float, float, float], ...]; mirror: bool = True

PATTERNS = (
    Pattern("light_short_pulse", ((0.5,0.9,0.25),)), Pattern("medium_pulse", ((0.5,1.5,0.5),)),
    Pattern("full_long_hold", ((0.5,2.5,1.0),)), Pattern("small_sustained", ((0.5,2.5,0.25),)),
    Pattern("move_stop_move", ((0.5,1.5,0.6),(3.5,4.5,0.6))),
    Pattern("commanded_reversal", ((0.5,2.0,0.6),(2.0,3.5,-0.6))),
    Pattern("repeated_pulse_release", ((0.5,1.1,0.5),(1.9,2.5,0.5),(3.3,3.9,0.5))),
    Pattern("gradual_taper", ((0.5,1.5,0.8),(1.5,2.0,0.5),(2.0,2.5,0.25))),
)

def _arr(rows, key): return np.asarray([float(r[key]) for r in rows])
def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)

def _scenarios():
    for pattern in PATTERNS:
        for sign, direction in ((1,"positive"),(-1,"negative")) if pattern.mirror else ((1,"positive"),):
            timeline=tuple((a,b,sign*x) for a,b,x in pattern.timeline)
            yield pattern.key, sign, LoiterScenarioConfig(name=f"{pattern.key}_{direction}",duration_s=10.0,initial_z=1,target_z=1,capture_current_target=True,stick_timeline=timeline,notes="final event-aware LOITER regression")

def _audit(run):
    rows, scenario = run.rows, run.scenario
    t,vx,target,count=_arr(rows,"time"),_arr(rows,"vx"),_arr(rows,"target_x"),_arr(rows,"target_capture_count")
    definition=ScenarioDefinition(scenario.name,scenario,"loiter",1,0.5,False,False)
    metrics=compute_metrics(definition,run,quick=False)
    metrics.update(_moving_mass_metrics(run))
    events=[]; failures=[]
    segments=list(scenario.stick_timeline)
    metrics.update({"premature_pause":False,"second_acceleration_lobe":False,"shaped_sign_reversal":False,"early_velocity_reversal":False,"target_jump":False,"capture_count_monotonic":bool(np.all(np.diff(count)>=0.0)),"final_cumulative_capture_count":int(count[-1])})
    for i, (start,end,command) in enumerate(segments):
        # An intentional direct reversal has no release boundary.
        next_start=segments[i+1][0] if i+1<len(segments) else math.inf
        release = end < next_start - 1e-9
        active=(t>=start)&(t<end)
        # A segment that starts immediately after an opposite-sign segment is
        # an intentional commanded reversal, not a release overshoot.
        previous = segments[i-1][2] if i else 0.0
        if previous * command >= 0.0 and np.any(active) and np.any(vx[active]*command < -0.03): failures.append(f"early_actual_velocity_reversal@{start:.2f}"); metrics["early_velocity_reversal"]=True
        if previous * command >= 0.0:
            normalized=[{**row,"vx":float(row["vx"])*(1 if command>0 else -1),"x_error":float(row["x_error"])*(1 if command>0 else -1),"x":float(row["x"])*(1 if command>0 else -1)} for row in np.asarray(rows,dtype=object)[active]]
            if detect_premature_pause(normalized) is not None: failures.append(f"premature_pause@{start:.2f}"); metrics["premature_pause"]=True
        if release:
            later=(t>=end)&(t<next_start)
            increments=int(count[np.flatnonzero(later)[-1]]-count[np.flatnonzero(t<end)[-1]]) if np.any(later) and np.any(t<end) else 0
            events.append({"event":"release","time_s":end,"command":command,"capture_increments":increments})
            # Inter-pulse windows intentionally restart motion before the
            # capture criteria complete; require exactly one increment only
            # for a completed final release or an explicitly long stop.
            completed = math.isinf(next_start) or next_start - end >= 1.5
            if completed and increments != 1: failures.append(f"capture_increment@{end:.2f}:{increments}")
            shaped=_arr(rows,"shaped_desired_vx")[later]
            if shaped.size>1 and np.any(shaped*command < -1e-4): failures.append(f"shaped_sign_reversal@{end:.2f}"); metrics["shaped_sign_reversal"]=True
            # Start the validated detector only after the first post-release
            # motion peak; otherwise its initial near-zero samples precede the
            # normal shaped response and create a false second-lobe report.
            post=np.flatnonzero(t>=end)
            if completed:
                window=(t>=end)&(t<next_start)
                normalized=[{**row,"vx":float(row["vx"])*(1 if command>0 else -1)} for row in np.asarray(rows,dtype=object)[window]]
                if normalized:
                    nt=_arr(normalized,"time"); nv=_arr(normalized,"vx")
                    peak=int(np.argmax(nv))
                    # Do not let pre-response near-zero samples count as a
                    # pause: start after the first same-direction motion peak.
                    if nv[peak]>=0.10 and second_acceleration_lobe_after_full_pause(normalized,release_time_s=float(nt[peak])):
                        failures.append(f"second_acceleration_lobe@{end:.2f}"); metrics["second_acceleration_lobe"]=True
    if not metrics["capture_count_monotonic"]: failures.append("capture_count_non_monotonic")
    if np.any(np.abs(np.diff(target))>0.02): failures.append("target_jump"); metrics["target_jump"]=True
    if np.any(~np.isfinite(_arr(rows,"x"))) or any(str(r.get("crash_reason","")) for r in rows): failures.append("invalid_or_crash")
    if metrics["meaningful_vane_sign_change_count"]>CHATTER_THRESHOLDS["max_meaningful_sign_changes"] or metrics["vane_total_variation_per_second_deg_s"]>CHATTER_THRESHOLDS["max_total_variation_per_second_deg_s"] or metrics["tail_high_frequency_vane_energy_deg2"]>CHATTER_THRESHOLDS["max_tail_high_frequency_energy_deg2"]: failures.append("vane_chatter")
    for key,reason in (("vane_saturation_percent","vane_saturation"),("servo_rate_saturation_percent","servo_rate_saturation"),("mixer_saturation_percent","mixer_saturation")):
        if metrics[key]>HARD_GATE_THRESHOLDS[key]: failures.append(reason)
    if metrics["meaningful_moving_mass_direction_change_count"]>30 or metrics["moving_mass_total_travel_per_second_m_s"]>0.5 or metrics["tail_high_frequency_moving_mass_energy_m2"]>2.5e-5: failures.append("moving_mass_chatter")
    for observed,limit,reason in ((metrics["moving_mass_max_abs_offset_m"],0.05,"moving_mass_offset_limit"),(metrics["moving_mass_max_abs_velocity_m_s"],0.2,"moving_mass_rate_limit"),(metrics["moving_mass_max_abs_acceleration_m_s2"],1.0,"moving_mass_acceleration_limit")):
        if observed>limit+1e-9: failures.append(reason)
    for name, limits in MOVING_MASS_LIMITER_HARD_GATES.items():
        if metrics[f"moving_mass_{name}_duty_percent"]>limits["max_duty_percent"] or metrics[f"moving_mass_{name}_longest_continuous_duration_s"]>limits["max_continuous_duration_s"]: failures.append(f"excessive_{name}")
    return metrics, events, failures

def run(output_dir: Path=OUT):
    output_dir.mkdir(parents=True,exist_ok=True); summaries=[]; audits=[]; digests=[]
    for rerun in range(2):
        rows_out=[]; event_out=[]
        for pattern,direction,scenario in _scenarios():
            for variant,gain in VARIANTS:
                effective=replace(scenario,moving_mass_enabled=True,moving_mass_target_m=0.0,moving_mass_assist_gain_m_per_Nm=gain)
                result=run_headless_loiter(PROFILE,effective,controller_overrides={**FIXED_CONTROLLER,"enable_noise":False,"random_seed":0})
                metrics,events,failures=_audit(result)
                executed_gain=float(result.scenario.moving_mass_assist_gain_m_per_Nm)
                if executed_gain != gain: failures.append("executed_gain_mismatch")
                if gain == 0.0 and any(abs(_arr(result.rows,key)).max() != 0.0 for key in ("moving_mass_offset_m","moving_mass_target_m","moving_mass_velocity_m_s")): failures.append("vane_only_not_locked")
                if gain > 0.0 and not any(abs(_arr(result.rows,key)).max() > 0.0 for key in ("moving_mass_offset_m","moving_mass_target_m")): failures.append("assist_no_moving_mass_response")
                metric_direction=metrics.pop("direction", None)
                row={"pattern":pattern,"variant":variant,"gain":executed_gain,"metric_direction":metric_direction,"passed":not failures,"failures":"; ".join(failures),**metrics,"direction":"positive" if direction>0 else "negative"}; rows_out.append(row)
                event_out.extend({"pattern":pattern,"direction":"positive" if direction>0 else "negative","variant":variant,"final_cumulative_capture_count":metrics["final_cumulative_capture_count"],**e} for e in events)
                if rerun==0 and (not row["passed"] or pattern in ("medium_pulse","commanded_reversal")): save_loiter_timeseries(result.rows,output_dir/"timeseries"/f"{pattern}_{direction}_{variant}.csv")
                if rerun == 0 and direction > 0 and variant == "vane_only":
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    figure, axis = plt.subplots(figsize=(8, 3))
                    axis.plot(_arr(result.rows,"time"), _arr(result.rows,"vx"), label="actual vx")
                    axis.plot(_arr(result.rows,"time"), _arr(result.rows,"shaped_desired_vx"), label="shaped desired vx")
                    axis.set(title=pattern, xlabel="time (s)", ylabel="m/s"); axis.grid(alpha=.3); axis.legend()
                    (output_dir/"plots").mkdir(parents=True, exist_ok=True); figure.savefig(output_dir/"plots"/f"{pattern}.png",dpi=140); plt.close(figure)
        digests.append(_sha256_bytes(_canonical_json(rows_out).encode()))
        if rerun==0: summaries,event_audit=rows_out,event_out
    if len(set(digests))!=1: raise RuntimeError("non-deterministic regression")
    _write(output_dir/"scenario_metrics.csv",summaries); _write(output_dir/"event_capture_audit.csv",event_audit)
    (output_dir/"scenario_specification.json").write_text(json.dumps({"patterns":[asdict(p) for p in PATTERNS],"variants":VARIANTS,"controller":FIXED_CONTROLLER},indent=2),encoding="utf-8")
    passed=all(r["passed"] for r in summaries)
    lines=["# Final LOITER regression", "", f"Passed: **{passed}**. Cases: {len(summaries)}.", "", "| pattern | direction | variant | executed gain | capture count | pause / second lobe / shaped sign / early reversal | vane RMS | mass max offset/rate/accel |", "| --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    for row in summaries:
        lines.append(f"| {row['pattern']} | {row['direction']} | {row['variant']} | {float(row['gain']):.5f} | {int(float(row.get('target_capture_count',0)))} | {row['failures'] or 'pass'} | {float(row.get('vane_command_rms_deg',0)):.4f} | {float(row['moving_mass_max_abs_offset_m']):.5f} / {float(row['moving_mass_max_abs_velocity_m_s']):.5f} / {float(row['moving_mass_max_abs_acceleration_m_s2']):.5f} |")
    (output_dir/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (output_dir/"deterministic.json").write_text(json.dumps({"digests":digests,"passed":passed},indent=2),encoding="utf-8")
    artifacts={p.relative_to(output_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.rglob("*") if p.is_file() and p.name!="manifest.json"}
    (output_dir/"manifest.json").write_text(json.dumps({"passed":passed,"artifacts":artifacts},indent=2),encoding="utf-8")
    return passed,summaries

if __name__ == "__main__":
    ok,_=run(); print(json.dumps({"passed":ok})); raise SystemExit(0 if ok else 2)

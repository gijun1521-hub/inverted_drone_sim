from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


@dataclass(frozen=True)
class EquivalentPitchVaneCommand:
    """Current 2D pitch-axis equivalent vane command."""

    moment_cmd: float
    vane_angle_cmd_deg: float
    vane_angle_actual_deg: float = 0.0
    angle_saturated: bool = False
    rate_saturated: bool = False
    authority_limited: bool = False
    mixer_saturated: bool = False

    @property
    def saturated(self) -> bool:
        return bool(self.angle_saturated or self.rate_saturated or self.authority_limited or self.mixer_saturated)


@dataclass(frozen=True)
class FourVaneCommand:
    front_vane_deg: float
    right_vane_deg: float
    rear_vane_deg: float
    left_vane_deg: float

    @property
    def max_abs_vane_deg(self) -> float:
        return max(
            abs(self.front_vane_deg),
            abs(self.right_vane_deg),
            abs(self.rear_vane_deg),
            abs(self.left_vane_deg),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "front_vane_deg": float(self.front_vane_deg),
            "right_vane_deg": float(self.right_vane_deg),
            "rear_vane_deg": float(self.rear_vane_deg),
            "left_vane_deg": float(self.left_vane_deg),
        }


@dataclass(frozen=True)
class FourVaneMixerInput:
    roll_moment_cmd: float = 0.0
    pitch_moment_cmd: float = 0.0
    yaw_moment_cmd: float = 0.0
    thrust_cmd: float = 0.0


@dataclass(frozen=True)
class FourVaneMixerOutput:
    command: FourVaneCommand
    requested_roll_moment: float
    requested_pitch_moment: float
    requested_yaw_moment: float
    thrust_cmd: float
    equivalent_2d_moment: float
    roll_saturated: bool
    pitch_saturated: bool
    yaw_saturated: bool
    mode_2d_equivalent: bool
    notes: str = ""

    @property
    def saturated(self) -> bool:
        return bool(self.roll_saturated or self.pitch_saturated or self.yaw_saturated)

    @property
    def max_abs_vane_cmd_deg(self) -> float:
        return self.command.max_abs_vane_deg

    def diagnostic_row(self) -> dict[str, float | bool | str]:
        return {
            "requested_roll_moment": float(self.requested_roll_moment),
            "requested_pitch_moment": float(self.requested_pitch_moment),
            "requested_yaw_moment": float(self.requested_yaw_moment),
            "equivalent_2d_moment": float(self.equivalent_2d_moment),
            "thrust_cmd": float(self.thrust_cmd),
            "front_vane_cmd_deg": float(self.command.front_vane_deg),
            "right_vane_cmd_deg": float(self.command.right_vane_deg),
            "rear_vane_cmd_deg": float(self.command.rear_vane_deg),
            "left_vane_cmd_deg": float(self.command.left_vane_deg),
            "max_abs_vane_cmd_deg": float(self.max_abs_vane_cmd_deg),
            "roll_saturated": bool(self.roll_saturated),
            "pitch_saturated": bool(self.pitch_saturated),
            "yaw_saturated": bool(self.yaw_saturated),
            "saturated": bool(self.saturated),
            "mode_2d_equivalent": bool(self.mode_2d_equivalent),
            "notes": self.notes,
        }

    def diagnostic_text(self) -> str:
        mode = "2D equivalent" if self.mode_2d_equivalent else "conceptual 4-vane"
        return (
            f"{mode}: roll={self.requested_roll_moment:.3f}, pitch={self.requested_pitch_moment:.3f}, "
            f"yaw={self.requested_yaw_moment:.3f}, vanes="
            f"front/right/rear/left {self.command.front_vane_deg:.2f}/"
            f"{self.command.right_vane_deg:.2f}/{self.command.rear_vane_deg:.2f}/"
            f"{self.command.left_vane_deg:.2f} deg, saturated={int(self.saturated)}"
        )


class ConceptualFourVaneMixer:
    """Conceptual four-vane command mapper for architecture prep, not 3D physics."""

    def __init__(self, max_vane_angle_deg: float, moment_to_vane_deg: float = 1.0):
        self.max_vane_angle_deg = float(max_vane_angle_deg)
        self.moment_to_vane_deg = float(moment_to_vane_deg)

    def mix(self, mixer_input: FourVaneMixerInput) -> FourVaneMixerOutput:
        roll = float(mixer_input.roll_moment_cmd) * self.moment_to_vane_deg
        pitch = float(mixer_input.pitch_moment_cmd) * self.moment_to_vane_deg
        yaw = float(mixer_input.yaw_moment_cmd) * self.moment_to_vane_deg
        raw = FourVaneCommand(
            front_vane_deg=pitch + yaw,
            right_vane_deg=roll + yaw,
            rear_vane_deg=-pitch + yaw,
            left_vane_deg=-roll + yaw,
        )
        command = FourVaneCommand(
            front_vane_deg=_clamp(raw.front_vane_deg, self.max_vane_angle_deg),
            right_vane_deg=_clamp(raw.right_vane_deg, self.max_vane_angle_deg),
            rear_vane_deg=_clamp(raw.rear_vane_deg, self.max_vane_angle_deg),
            left_vane_deg=_clamp(raw.left_vane_deg, self.max_vane_angle_deg),
        )
        return FourVaneMixerOutput(
            command=command,
            requested_roll_moment=float(mixer_input.roll_moment_cmd),
            requested_pitch_moment=float(mixer_input.pitch_moment_cmd),
            requested_yaw_moment=float(mixer_input.yaw_moment_cmd),
            thrust_cmd=float(mixer_input.thrust_cmd),
            equivalent_2d_moment=float(mixer_input.pitch_moment_cmd),
            roll_saturated=abs(roll) > self.max_vane_angle_deg,
            pitch_saturated=abs(pitch) > self.max_vane_angle_deg,
            yaw_saturated=raw.max_abs_vane_deg > self.max_vane_angle_deg,
            mode_2d_equivalent=False,
            notes="Conceptual mixer only; not full 3D SingleCopter physics.",
        )


def equivalent_pitch_vane_to_four_vane(
    command: EquivalentPitchVaneCommand,
    *,
    max_vane_angle_deg: float | None = None,
) -> FourVaneMixerOutput:
    """Map the current 2D equivalent pitch vane to front/rear differential vanes."""

    limit = abs(float(max_vane_angle_deg)) if max_vane_angle_deg is not None else None
    raw_pitch = float(command.vane_angle_cmd_deg)
    pitch = _clamp(raw_pitch, limit) if limit is not None else raw_pitch
    four_vane = FourVaneCommand(
        front_vane_deg=pitch,
        right_vane_deg=0.0,
        rear_vane_deg=-pitch,
        left_vane_deg=0.0,
    )
    adapter_saturated = limit is not None and abs(raw_pitch) > limit
    return FourVaneMixerOutput(
        command=four_vane,
        requested_roll_moment=0.0,
        requested_pitch_moment=float(command.moment_cmd),
        requested_yaw_moment=0.0,
        thrust_cmd=0.0,
        equivalent_2d_moment=float(command.moment_cmd),
        roll_saturated=False,
        pitch_saturated=bool(command.saturated or adapter_saturated),
        yaw_saturated=False,
        mode_2d_equivalent=True,
        notes="2D equivalent pitch adapter; roll and yaw channels are placeholders.",
    )

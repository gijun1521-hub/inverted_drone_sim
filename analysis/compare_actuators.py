from __future__ import annotations

from dataclasses import dataclass

try:
    from ..config import MovingMassConfig, RigidBodyConfig
except ImportError:  # pragma: no cover
    from config import MovingMassConfig, RigidBodyConfig


@dataclass(frozen=True)
class AuthorityComparison:
    vane_moment: float
    moving_mass_reaction_moment: float
    moving_mass_cg_offset_moment: float
    hybrid_total_moment: float


def compare_authority(rb: RigidBodyConfig | None = None, mm: MovingMassConfig | None = None) -> AuthorityComparison:
    rb = rb or RigidBodyConfig()
    mm = mm or MovingMassConfig()
    vane = abs(rb.k_moment) * rb.hover_thrust * rb.vane_angle_max
    reaction = abs(mm.I_moving_about_hinge * mm.q_accel_limit)
    cg_offset = mm.thrust * (mm.m_moving / mm.m_total) * abs(mm.mass_center_offset_body[1])
    return AuthorityComparison(vane, reaction, cg_offset, vane + reaction + cg_offset)

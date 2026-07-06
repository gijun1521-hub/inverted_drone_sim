from __future__ import annotations

from pathlib import Path

try:
    from .analysis.authority_maps import generate_authority_maps
    from .analysis.compare_actuators import compare_authority
    from .analysis.nondimensional import save_nondimensional_summary
except ImportError:  # pragma: no cover
    from analysis.authority_maps import generate_authority_maps
    from analysis.compare_actuators import compare_authority
    from analysis.nondimensional import save_nondimensional_summary


def main() -> None:
    out = Path("results/analysis")
    summary = save_nondimensional_summary(out / "nondimensional_summary.csv")
    maps = generate_authority_maps(out)
    comparison = compare_authority()
    report = out / "authority_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Authority Analysis\n\n"
        "These are analytical trend checks, not experimentally calibrated results.\n\n"
        f"- Vane moment proxy: {comparison.vane_moment:.4f} N m\n"
        f"- Moving-mass reaction proxy: {comparison.moving_mass_reaction_moment:.4f} N m\n"
        f"- Moving-mass CG-offset proxy: {comparison.moving_mass_cg_offset_moment:.4f} N m\n"
        f"- Hybrid proxy: {comparison.hybrid_total_moment:.4f} N m\n",
        encoding="utf-8",
    )
    print(f"saved: {summary}")
    for path in maps:
        print(f"saved: {path}")
    print(f"saved: {report}")


if __name__ == "__main__":
    main()

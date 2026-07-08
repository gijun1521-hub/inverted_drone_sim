import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RigidBodyConfig
from mixers import (
    ConceptualFourVaneMixer,
    EquivalentPitchVaneCommand,
    FourVaneCommand,
    FourVaneMixerInput,
    equivalent_pitch_vane_to_four_vane,
)
from singlecopter_mixer import SingleCopterMixer


class FourVaneMixerPrepTests(unittest.TestCase):
    def test_four_vane_command_dataclass_can_be_created(self):
        command = FourVaneCommand(front_vane_deg=1.0, right_vane_deg=2.0, rear_vane_deg=-1.0, left_vane_deg=-2.0)

        self.assertEqual(command.front_vane_deg, 1.0)
        self.assertEqual(command.as_dict()["left_vane_deg"], -2.0)
        self.assertEqual(command.max_abs_vane_deg, 2.0)

    def test_zero_input_produces_zero_vane_outputs(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=20.0)

        output = mixer.mix(FourVaneMixerInput())

        self.assertEqual(output.command, FourVaneCommand(0.0, 0.0, 0.0, 0.0))
        self.assertFalse(output.saturated)

    def test_positive_pitch_command_produces_front_rear_differential(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=20.0, moment_to_vane_deg=10.0)

        output = mixer.mix(FourVaneMixerInput(pitch_moment_cmd=0.5))

        self.assertEqual(output.command.front_vane_deg, 5.0)
        self.assertEqual(output.command.rear_vane_deg, -5.0)
        self.assertEqual(output.command.right_vane_deg, 0.0)
        self.assertEqual(output.command.left_vane_deg, 0.0)

    def test_negative_pitch_command_flips_front_rear_signs(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=20.0, moment_to_vane_deg=10.0)

        output = mixer.mix(FourVaneMixerInput(pitch_moment_cmd=-0.5))

        self.assertEqual(output.command.front_vane_deg, -5.0)
        self.assertEqual(output.command.rear_vane_deg, 5.0)

    def test_roll_and_yaw_channels_are_handled_conceptually(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=20.0, moment_to_vane_deg=10.0)

        roll = mixer.mix(FourVaneMixerInput(roll_moment_cmd=0.5))
        yaw = mixer.mix(FourVaneMixerInput(yaw_moment_cmd=0.5))

        self.assertEqual(roll.command.right_vane_deg, 5.0)
        self.assertEqual(roll.command.left_vane_deg, -5.0)
        self.assertEqual(roll.command.front_vane_deg, 0.0)
        self.assertEqual(roll.command.rear_vane_deg, 0.0)
        self.assertEqual(yaw.command, FourVaneCommand(5.0, 5.0, 5.0, 5.0))

    def test_saturation_clamps_vane_commands(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=12.0, moment_to_vane_deg=10.0)

        output = mixer.mix(FourVaneMixerInput(pitch_moment_cmd=2.0))

        self.assertEqual(output.command.front_vane_deg, 12.0)
        self.assertEqual(output.command.rear_vane_deg, -12.0)
        self.assertTrue(output.pitch_saturated)
        self.assertTrue(output.saturated)

    def test_equivalent_2d_adapter_maps_pitch_to_front_rear_only(self):
        equivalent = EquivalentPitchVaneCommand(moment_cmd=-0.2, vane_angle_cmd_deg=-7.5, vane_angle_actual_deg=-6.0)

        output = equivalent_pitch_vane_to_four_vane(equivalent)

        self.assertTrue(output.mode_2d_equivalent)
        self.assertEqual(output.equivalent_2d_moment, -0.2)
        self.assertEqual(output.command.front_vane_deg, -7.5)
        self.assertEqual(output.command.rear_vane_deg, 7.5)
        self.assertEqual(output.command.right_vane_deg, 0.0)
        self.assertEqual(output.command.left_vane_deg, 0.0)

    def test_equivalent_2d_adapter_can_report_clamped_display_commands(self):
        equivalent = EquivalentPitchVaneCommand(moment_cmd=0.4, vane_angle_cmd_deg=30.0)

        output = equivalent_pitch_vane_to_four_vane(equivalent, max_vane_angle_deg=20.0)

        self.assertEqual(output.command.front_vane_deg, 20.0)
        self.assertEqual(output.command.rear_vane_deg, -20.0)
        self.assertTrue(output.pitch_saturated)

    def test_diagnostics_include_future_logging_fields(self):
        mixer = ConceptualFourVaneMixer(max_vane_angle_deg=20.0, moment_to_vane_deg=10.0)

        row = mixer.mix(FourVaneMixerInput(roll_moment_cmd=0.1, pitch_moment_cmd=0.2, yaw_moment_cmd=0.0, thrust_cmd=3.0)).diagnostic_row()

        self.assertEqual(row["requested_roll_moment"], 0.1)
        self.assertEqual(row["requested_pitch_moment"], 0.2)
        self.assertEqual(row["requested_yaw_moment"], 0.0)
        self.assertEqual(row["equivalent_2d_moment"], 0.2)
        self.assertEqual(row["thrust_cmd"], 3.0)
        self.assertIn("front_vane_cmd_deg", row)
        self.assertIn("mode_2d_equivalent", row)
        self.assertIn("front/right/rear/left", mixer.mix(FourVaneMixerInput(pitch_moment_cmd=0.2)).diagnostic_text())

    def test_current_2d_mixer_output_still_feeds_adapter_without_recomputing_physics(self):
        cfg = RigidBodyConfig()
        current_mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)
        current = current_mixer.mix(desired_moment=-0.1, thrust=cfg.hover_thrust)
        equivalent = EquivalentPitchVaneCommand(
            moment_cmd=current.requested_moment,
            vane_angle_cmd_deg=current.vane_angle_cmd * 180.0 / 3.141592653589793,
            angle_saturated=current.angle_saturated,
            authority_limited=current.authority_limited,
            mixer_saturated=current.saturated,
        )

        output = equivalent_pitch_vane_to_four_vane(equivalent)

        self.assertAlmostEqual(output.requested_pitch_moment, current.requested_moment)
        self.assertAlmostEqual(output.command.front_vane_deg, equivalent.vane_angle_cmd_deg)
        self.assertAlmostEqual(output.command.rear_vane_deg, -equivalent.vane_angle_cmd_deg)


if __name__ == "__main__":
    unittest.main()

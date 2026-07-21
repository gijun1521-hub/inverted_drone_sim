from __future__ import annotations
import unittest
from dataclasses import replace
from analysis.final_loiter_regression import VARIANTS, _scenarios

class FinalLoiterRegressionTests(unittest.TestCase):
    def test_variants_have_distinct_executed_gains(self):
        scenario=next(_scenarios())[2]
        executed=[replace(scenario,moving_mass_enabled=True,moving_mass_target_m=0.0,moving_mass_assist_gain_m_per_Nm=g).moving_mass_assist_gain_m_per_Nm for _,g in VARIANTS]
        self.assertEqual(executed,[0.0,0.0415])
        self.assertNotEqual(*executed)

    def test_reversal_and_interrupted_release_are_declared_patterns(self):
        keys={item[0] for item in _scenarios()}
        self.assertIn("commanded_reversal",keys)
        self.assertIn("repeated_pulse_release",keys)

if __name__=="__main__": unittest.main()

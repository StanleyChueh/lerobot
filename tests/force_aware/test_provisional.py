"""CPU contract tests using synthetic telemetry only."""

import ast
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lerobot.policies.common.vla_utils import pad_vector
from lerobot.utils.constants import OBS_STATE
from tools.force_aware.provisional import Aligned, Contract, Packet, TrainNormalizer, align, preview_state


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        # Fixture units are invented for tests; these are not a hardware declaration.
        self.contract = Contract(
            tuple(f"{arm}J{i}.pos" for i in range(1, 9) for arm in ("R", "L")),
            ("rad",) * 16,
            ("rad/s",) * 16,
            ("N*m",) * 16,
        )
        self.packet = Packet("sha/0", 1.0, 1.01, self.contract, tuple(range(16)), tuple(range(16, 32)))

    def join(self, packets: list[Packet], **kwargs: Any) -> Aligned:
        opts = {
            "episode": "sha/0",
            "observation_s": 1.02,
            "action_s": 1.03,
            "target": self.contract,
            "max_age_s": 0.05,
            "max_action_delay_s": 0.04,
        }
        return align(packets, **(opts | kwargs))

    def test_reordering_and_units(self) -> None:
        c = replace(self.contract, names=self.contract.names[::-1])
        p = replace(
            self.packet, contract=c, velocity=self.packet.velocity[::-1], effort=self.packet.effort[::-1]
        )
        np.testing.assert_array_equal(self.join([p]).values, np.arange(32))
        with self.assertRaisesRegex(ValueError, "calibration"):
            self.join([replace(p, contract=replace(c, velocity_units=("deg/s",) * 16))])
        with self.assertRaises(ValueError):
            replace(c, names=c.names[:-1])

    def test_causal_alignment_and_delay(self) -> None:
        future = replace(self.packet, acquired_s=1.025, available_s=1.026)
        late = replace(self.packet, acquired_s=1.015, available_s=1.04)
        np.testing.assert_array_equal(self.join([future, late, self.packet]).values, np.arange(32))
        self.assertFalse(self.join([late]).valid)
        with self.assertRaises(ValueError):
            self.join([self.packet], action_s=1.01)
        with self.assertRaises(ValueError):
            self.join([self.packet], action_s=1.2)

    def test_missing_stale_and_episode_isolation(self) -> None:
        for packets in ([], [replace(self.packet, episode="sha/1")], [self.packet]):
            r = self.join(packets, observation_s=1.2, action_s=1.21)
            self.assertFalse(r.valid)
            np.testing.assert_array_equal(r.values, np.zeros(32))
        with self.assertRaises(ValueError):
            self.join([self.packet, self.packet])
        with self.assertRaises(ValueError):
            self.join([replace(self.packet, effort=(float("nan"),) * 16)])

    def test_train_only_statistics(self) -> None:
        rows = {"sha/0": np.ones((2, 32)), "sha/1": np.full((2, 32), 3.0)}
        norm = TrainNormalizer.fit(rows, train_ids=set(rows), test_ids={"sha/2"})
        np.testing.assert_array_equal(norm.mean, np.full(32, 2.0))
        np.testing.assert_array_equal(norm.scale, np.ones(32))
        for train, test in [(set(rows), {"sha/0"}), ({"sha/0"}, {"sha/2"})]:
            with self.assertRaises(ValueError):
                TrainNormalizer.fit(rows, train_ids=train, test_ids=test)
        with self.assertRaises(ValueError):
            norm.transform(self.join([]))

    def test_feature_width_and_baseline_preservation(self) -> None:
        state = np.arange(16.0)
        baseline = preview_state(state)
        np.testing.assert_array_equal(baseline, state)
        self.assertFalse(np.shares_memory(state, baseline))
        with self.assertRaises(ValueError):
            preview_state(state, self.join([self.packet]).values)
        self.assertEqual(preview_state(state, self.join([self.packet]).values, max_state_dim=48).shape, (48,))

    def test_legacy_state_projection_round_trip(self) -> None:
        # Real SmolVLA prepare_state method, extracted without constructing/downloading the VLM.
        path = Path("src/lerobot/policies/smolvla/modeling_smolvla.py")
        tree = ast.parse(path.read_text())
        method = next(
            n
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_state"
        )
        namespace = {"pad_vector": pad_vector, "OBS_STATE": OBS_STATE}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        raw = torch.arange(16.0).reshape(1, 16)
        padded = namespace["prepare_state"](
            SimpleNamespace(config=SimpleNamespace(max_state_dim=32)), {OBS_STATE: raw}
        )
        torch.testing.assert_close(padded[:, :16], raw)
        torch.testing.assert_close(padded[:, 16:], torch.zeros(1, 16))
        old = torch.nn.Linear(32, 8)
        restored = torch.nn.Linear(32, 8)
        restored.load_state_dict(old.state_dict(), strict=True)
        torch.testing.assert_close(old(padded), restored(padded), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

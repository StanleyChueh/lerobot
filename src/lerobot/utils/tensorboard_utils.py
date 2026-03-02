#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path

from termcolor import colored


class TensorBoardLogger:
    """A helper class to log metrics and videos using TensorBoard.

    Mirrors the interface of :class:`lerobot.rl.wandb_utils.WandBLogger` so the two
    loggers can be used interchangeably from the training loop.

    Args:
        cfg: The full :class:`~lerobot.configs.train.TrainPipelineConfig`.  The
            TensorBoard event files are written to
            ``cfg.output_dir / cfg.tensorboard.log_dir``.
    """

    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(cfg.output_dir) / cfg.tensorboard.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._env_fps = cfg.env.fps if cfg.env else 4
        logging.info(
            colored("Logs will be written to TensorBoard.", "blue", attrs=["bold"])
        )
        logging.info(
            f"Launch TensorBoard with: {colored(f'tensorboard --logdir {log_dir}', 'yellow', attrs=['bold'])}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_dict(
        self,
        d: dict,
        step: int,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """Log a dictionary of scalar values.

        Args:
            d: Dictionary mapping metric names to scalar values.
            step: Global training step used as the x-axis value.
            mode: Either ``"train"`` or ``"eval"``. Metrics are tagged
                  ``<mode>/<key>`` in TensorBoard.
            custom_step_key: Unused — kept for API compatibility with
                :class:`~lerobot.rl.wandb_utils.WandBLogger`.
        """
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got '{mode}'")

        for k, v in d.items():
            if not isinstance(v, (int, float)):
                logging.warning(
                    f'TensorBoard logging of key "{k}" was ignored as its type '
                    f'"{type(v)}" is not a scalar.'
                )
                continue
            self._writer.add_scalar(f"{mode}/{k}", v, global_step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Log a video file to TensorBoard.

        The video is decoded with :func:`torchvision.io.read_video` and written
        using :meth:`~torch.utils.tensorboard.SummaryWriter.add_video`.

        Args:
            video_path: Path to an MP4 (or other torchvision-readable) video file.
            step: Global training step.
            mode: Either ``"train"`` or ``"eval"``.
        """
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got '{mode}'")

        try:
            import torchvision.io as tvio

            # read_video returns (video, audio, info); video shape: (T, H, W, C) uint8
            video_tensor, _, _ = tvio.read_video(video_path, output_format="TCHW", pts_unit="sec")
            # SummaryWriter.add_video expects (N, T, C, H, W)
            video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
            self._writer.add_video(
                f"{mode}/video",
                video_tensor,
                global_step=step,
                fps=self._env_fps,
            )
        except Exception as e:
            logging.warning(f"TensorBoard video logging failed for '{video_path}': {e}")

    def log_policy(self, checkpoint_dir: Path) -> None:
        """No-op — checkpoints are already saved to disk by the training loop.

        Kept for API compatibility with
        :class:`~lerobot.rl.wandb_utils.WandBLogger`.
        """
        pass

    def finish(self) -> None:
        """Flush and close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self._writer.flush()
        self._writer.close()

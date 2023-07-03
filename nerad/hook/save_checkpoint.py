import logging
from pathlib import Path
from typing import Any

import mitsuba as mi
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from nerad.model.config import SaveCheckpointConfig

logger = logging.getLogger(__name__)


class SaveCheckpointHook:
    def __init__(self, cfg: SaveCheckpointConfig):
        self.cfg = cfg

    def run(
        self,
        step: int,
        out_root: Path,
        optim: Optimizer,
        modules: dict[str, nn.Module],
        scheduler: _LRScheduler,
    ):
        cfg = self.cfg
        if step % cfg.step_size != 0 and not (cfg.first_step and step == 1):
            return

        out_dir = out_root / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.is_latest:
            file = "latest.ckpt"
        else:
            file = f"{step}.ckpt"

        ckpt = {
            "step": step,
        }

        if optim is not None:
            ckpt.update({
                "optim": optim.state_dict(),
                "modules": {
                    k: v.state_dict() for k, v in modules.items()
                }
            })
            for k, v in modules.items():
                for param in v.parameters():
                    if torch.any(torch.isnan(param)):
                        raise Exception('Nan encountered in checkpoint')

        if scheduler is not None:
            ckpt.update({
                "scheduler": scheduler.state_dict(),
            })


        logger.info(f"Save checkpoint: {out_dir / file}")
        torch.save(ckpt, out_dir / file)

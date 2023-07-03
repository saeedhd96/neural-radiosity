import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from mytorch.exp_lr import ExpLRScheduler

from nerad.integrator import registered_integrators
from nerad.loss import loss_registry
from nerad.model.config import (ComputeConfig, DatasetConfig,
                                ObjectConfig, RenderingConfig, TrainConfig)

from nerad.utils.dict_utils import inject_dict
from nerad.utils.io_utils import glob_sorted
from nerad.utils.json_utils import read_json
from nerad.utils.metric_utils import compute_metrics
from nerad.utils.sensor_utils import create_transforms
from nerad.utils.sparse_grid_utils import create_occupancy_cache

logger = logging.getLogger(__name__)


def configure_compute(cfg: ComputeConfig):
    logger.info(f"Set drjit flags to {cfg.dr_optimization_flags}")
    dr.set_flag(dr.JitFlag.LoopRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallRecord, cfg.dr_optimization_flags)
    dr.set_flag(dr.JitFlag.VCallOptimize, cfg.dr_optimization_flags)

    logger.info(f"Set torch detech anomaly to {cfg.torch_detect_anomaly}")
    torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

    logger.info(f"Seed everything with {cfg.seed}")
    seed_everything(cfg.seed)

    log_mitsuba_registration()


def create_integrator(
    cfg: RenderingConfig,
    scene: mi.Scene,
    scene_path: str,
    extra_config: dict[str, Any] = None,
    post_init_injection: dict[str, Any] = None,
    kwargs_injection: dict[str, Any] = None,
):
    mi_dict = {
        "type": cfg.integrator,
    }
    if extra_config is not None:
        mi_dict.update(extra_config)

    integrator_config = OmegaConf.to_container(cfg.config, resolve=True)
    if len(integrator_config) > 0:
        if cfg.integrator in registered_integrators:
            mi_dict["config"] = {
                "type": "dict",
                **integrator_config,
            }
        else:
            mi_dict.update(integrator_config)

    logger.info(f"Integrator dict: {mi_dict}")
    integrator = mi.load_dict(mi_dict)

    _mitsuba_post_init(cfg.post_init, integrator, scene, scene_path ,post_init_injection, kwargs_injection)
    logger.info(f"Integrator: {integrator}")
    return integrator


def load_dataset(cfg: DatasetConfig, device: str):
    learned_modules: dict[str, nn.Module] = {}

    scene = mi.load_file(cfg.scene)

    if cfg.cameras is None:
        transforms = create_transforms(cfg.scene, cfg.n_views)
    else:
        transforms = read_json(cfg.cameras)
    n_views = min(cfg.n_views, len(transforms)) if cfg.n_views > 0 else len(transforms)

    logger.info(f"Load {n_views} views from {len(transforms)} views")

    images = None
    if cfg.cameras is not None:
        images = load_exr_files(Path(cfg.cameras).parent / "exr", n_views)

    # Handle old training
    cfg = OmegaConf.to_container(cfg, resolve=True)

    return scene, transforms, images, learned_modules


def load_exr_files(folder: Path, limit: int = 0):
    files = glob_sorted(folder, "*.exr")
    logger.info(f"Loading files from {folder}:\n" + ", ".join([f.name for f in files]))

    if limit <= 0:
        limit = len(files)

    return [
        mi.Bitmap(str(file)) for file in files[:limit]
    ]


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_mitsuba_registration():
    logger.info(f"Registered integrators: {' '.join(registered_integrators)}")


def create_loss_function(config: ObjectConfig, n_steps: int):
    return loss_registry.build(
        config.name,
        inject_dict(config.config, {"n_steps": n_steps})
    )


def _mitsuba_post_init(cfg: Union[dict, DictConfig], obj: Any, scene: mi.Scene, scene_path: str ,injection: dict[str, Any] = None, kwargs_injection: dict[str, Any] = None):
    # NOTE: for unknown reasons, torch module creation must
    # happen after mitsuba object contruction (not during).
    # Therefore, we have this post_init hack.

    if not hasattr(obj, "post_init"):
        return

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict)

    if "kwargs" in cfg:
        bbox = scene.bbox()

        kwargs_injection = kwargs_injection or {}
        kwargs_injection.update({
            "scene_min": bbox.min,
            "scene_max": bbox.max,
        })
        for key in cfg["kwargs"].keys():
            occupancy_cache = None
            if 'embedding' in key:
                if 'occupancy_cache' in cfg['kwargs'][key]:
                    logger.info(f'Creating occupancy cahce for ({key})')
                    occupancy_cache = create_occupancy_cache(scene, scene_path, cfg['kwargs'][key]['resolution'])
                    cfg['kwargs'][key]['occupancy_cache'] = occupancy_cache

        inject_dict(cfg["kwargs"], kwargs_injection)

    injection = injection or {}
    inject_dict(cfg, injection)

    obj.post_init(**cfg)


def prepare_learned_objects(
    scene: mi.Scene,
    integrator: mi.Integrator,
    learned_modules: dict[str, nn.Module],
    train_cfg: Optional[TrainConfig],
    ckpt_file: Optional[str],
    device: str,
) -> dict[str, Any]:
    result = {}

    params = mi.traverse(scene)

    integrator_params = mi.traverse(integrator)

    # Register learned integrator
    if isinstance(integrator, nn.Module):
        learned_modules["integrator"] = integrator

    # Remove learned modules without parameter
    learned_modules = {
        k: v for k, v in learned_modules.items() if len(list(v.parameters())) > 0
    }

    logger.info(
        "Learned modules:\n" + ",\n".join([f"{name}: {obj}" for name, obj in learned_modules.items()])
    )

    # Hard-coded: all possible key suffixes that requires dr.jit and mitsuba handling
    dr_grad_keys = [
        ".grad_activator",
    ]

    for key, obj in params.items():
        if any((key.endswith(s) for s in dr_grad_keys)):
            logger.info(f"dr.enable_grad: {key}")
            dr.enable_grad(obj)
            continue

    for key, obj in integrator_params.items():
        if key == "grad_activator":
            logger.info(f"dr.enable_grad: integrator {key}")
            dr.enable_grad(obj)
            continue

    # Create PyTorch optimizer
    torch_optimized_params = []
    for key, obj in learned_modules.items():
        logger.info(f"Trained with PyTorch: {key}")
        obj.to(device)
        torch_optimized_params += list(obj.parameters())

    torch_optim = None
    if train_cfg is not None and len(torch_optimized_params) > 0:
        torch_optim = torch.optim.Adam(torch_optimized_params, lr=train_cfg.learning_rate,
                                       betas=(train_cfg.beta_1, train_cfg.beta_2))

    logger.info(
        "Optimizer summary:\n"
        f"PyTorch: {torch_optim is not None} ({len(torch_optimized_params)})"
    )

    # LR scheduling
    torch_scheduler = None
    if train_cfg is not None and train_cfg.lr_decay_start >= 0:
        lr_scheduler_args = (train_cfg.lr_decay_start, train_cfg.lr_decay_rate,
                             train_cfg.lr_decay_steps, train_cfg.lr_decay_min_rate)
        if torch_optim is not None:
            torch_scheduler = ExpLRScheduler(torch_optim, *lr_scheduler_args)

    # Resume training
    result["step"] = 0
    if ckpt_file is not None:
        logger.info(f"Load checkpoint {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        last_step = ckpt["step"]
        result["step"] = last_step
        logger.info(f"Checkpoint step is {last_step}")

        if torch_optim is not None:
            logger.info("Load torch optim")
            torch_optim.load_state_dict(ckpt["optim"])

        for name, obj in learned_modules.items():
            logger.info(f"Load torch module {name}")
            obj.load_state_dict(ckpt["modules"][name])

        if torch_scheduler is not None:
            torch_scheduler.load_state_dict(ckpt["scheduler"])

    result.update({
        "params": params,
        "integrator_params": integrator_params,
        "learned_modules": learned_modules,
        "torch_optim": torch_optim,
        "torch_scheduler": torch_scheduler,
    })

    return result


def compute_output_metrics(
    name: str,
    outputs: list[mi.Bitmap],
    integrator: str,
    gt: dict[str, mi.Bitmap],
):
    gt = gt.get(name)
    if gt is None:
        return {}

    is_image = name == "image"
    names = [name]
    if is_image and "nerad" in integrator:
        names = ["lhs", "rhs"]
    assert len(names) == len(outputs)

    results = {}
    for name, pred in zip(names, outputs):
        metrics = compute_metrics(pred, gt)
        for key, value in metrics.items():
            results[f"{name}_{key}"] = value

    return results

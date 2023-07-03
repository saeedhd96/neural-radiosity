import logging
import os
from os.path import isfile
from pathlib import Path
from typing import Optional

import drjit as dr
import hydra
import mitsuba as mi
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from common import (compute_output_metrics, configure_compute, create_integrator, create_loss_function,
                    load_dataset, prepare_learned_objects)
from mytorch.exp_lr import ExpLR, ExpLRScheduler
from mytorch.pbar import ProgressBar, ProgressBarConfig
from mytorch.utils.profiling_utils import (counter_profiler, time_profiler,
                                           vram_profiler)
from nerad.hook.save_checkpoint import SaveCheckpointHook
from nerad.hook.save_image import SaveImageHook
from nerad.hook.validation import ValidationHook
from nerad.integrator.nerad import Nerad
from nerad.model.config import TrainConfig
from nerad.utils.debug_utils import check_gradients
from nerad.utils.json_utils import write_json
from nerad.utils.sensor_utils import create_sensor
import time

logger = logging.getLogger(__name__)
pbar_config = ProgressBarConfig(1, ["Residual"], True)


@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(cfg: TrainConfig = None):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # profiling flags
    time_profiler.enabled = cfg.profile_time
    time_profiler.synchronize_cuda = cfg.profile_time_sync_cuda
    vram_profiler.enabled = cfg.profile_vram
    counter_profiler.enabled = cfg.profile_counter

    configure_compute(cfg.compute)
    device = os.environ.get("TORCH_DEVICE", "cuda:0")

    out_root = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Output: {out_root}")

    if cfg.is_watchdog_init:
        print(f"watchdog:out_root:{out_root}")
        return

    scene, transforms, images, learned_modules = load_dataset(cfg.dataset, device)
    rendering = cfg.rendering

    integrator_injection = {}
    residual_loss_function = create_loss_function(cfg.residual_loss, cfg.n_steps)
    integrator_injection["residual_function"] = residual_loss_function
    loss_functions = [residual_loss_function]

    integrator_function_injection = {"device": device}
    integrator = create_integrator(
        rendering,
        scene,
        cfg.dataset.scene,
        post_init_injection=integrator_injection,
        kwargs_injection=integrator_function_injection,
    )

    learned_info = prepare_learned_objects(
        scene,
        integrator,
        learned_modules,
        cfg,
        find_latest_ckpt(out_root / "checkpoints") if cfg.resume else None,
        device,
    )

    start_step: int = learned_info["step"] + 1
    params: mi.SceneParameters = learned_info["params"]
    integrator_params: mi.SceneParameters = learned_info["integrator_params"]
    torch_optim: torch.optim.Adam = learned_info["torch_optim"]
    torch_scheduler: ExpLRScheduler = learned_info["torch_scheduler"]
    learned_modules: dict[str, nn.Module] = learned_info["learned_modules"]

    end_step = cfg.n_steps
    if end_step <= start_step:
        logger.warning(f"end_step ({end_step}) <= start_step ({start_step}), aborting")
        return

    # Validation hooks
    validation_hooks = [ValidationHook(val_cfg, rendering, scene, cfg.dataset.scene ,integrator_injection, integrator_function_injection)
                        for val_cfg in cfg.validation.values()]
    for hook in validation_hooks:
        if  "nerad" in hook.rendering.integrator:
            assert isinstance(hook.get_integrator(), Nerad)
            hook.get_integrator().network = integrator.network

    # Hooks for saving mitsuba images
    image_hooks: list[SaveImageHook] = []

    # Saving hooks
    saving_hooks = [SaveCheckpointHook(save_cfg) for save_cfg in cfg.saving.values()]

    # Tensorboard
    writer = SummaryWriter(out_root / "tensorboard")

    # Training loop
    time_profiler.start("training")
    logger.info(f"Training from step {start_step} to {end_step}")
    pbar = ProgressBar(pbar_config, end_step - start_step + 1)
    for step in range(start_step, end_step + 1):
        if torch_optim is not None:
            torch_optim.zero_grad()

        if not cfg.profile_time:
            torch.cuda.empty_cache()
            dr.flush_malloc_cache()

        vram_profiler.take_snapshot(f"{step}_start")

        for loss_function in loss_functions:
            loss_function.update_state(step - 1)

        time_profiler.start("forward")

        loss = dr.mean(integrator.compute_residual(scene, cfg.batch_size, step + 1))

        time_profiler.end("forward")



        # record loss values every several steps to reduce GPU-CPU comm
        loss_values = {}

        def record_loss_value(key, value):
            if step % cfg.update_loss_step_size != 0:
                return
            loss_values[key] = float(str(value[0]))


        record_loss_value("Residual", loss)

        vram_profiler.take_snapshot(f"{step}_forward")

        time_profiler.start("backward")

        dr.backward(loss, dr.ADFlag.ClearEdges)

        if torch_optim is not None:
            torch_optim.step()
            if torch_scheduler is not None:
                torch_scheduler.step()
                writer.add_scalar("torch_learning_rate", torch_optim.param_groups[0]["lr"], global_step=step)

        time_profiler.end("backward")

        vram_profiler.take_snapshot(f"{step}_backward")

        if cfg.check_gradients:
            for name, obj in learned_modules.items():
                logger.info(f"Check gradients of {name}")
                check_gradients(list(obj.parameters()))

        if cfg.clipgrads > 0:
            for name, obj in learned_modules.items():
                torch.nn.utils.clip_grad_norm_(obj.parameters(), cfg.clipgrads)

        # Save checkpoints
        for hook in saving_hooks:
            hook.run(step, out_root, torch_optim, learned_modules, torch_scheduler)

        # Validation
        val_view_idx = cfg.validation_view
        if images is not None:
            val_gt = {
                "image": images[val_view_idx],
            }
        start_val = time.time()
        for hook in validation_hooks:
            val_sensor = create_sensor(
                512,
                transforms[str(val_view_idx)],
            )
            val_outputs = hook.run(step, out_root, f"{val_view_idx:03d}", val_sensor)

            if val_outputs is None:
                continue
            if images is not None:
                val_metrics = compute_output_metrics(hook.cfg.name, val_outputs, rendering.integrator, val_gt)
                for key, value in val_metrics.items():
                    writer.add_scalar(f"metric/{key}", value, global_step=step)
                if len(val_metrics) > 0:
                    logger.info(
                        "Validation: " + ", ".join((f"{k}={v:.3f}" for k, v in val_metrics.items()))
                    )
        # Save extra images
        for hook in image_hooks:
            hook.run(step, out_root)

        end_val = time.time()
        validation_time=end_val-start_val
        writer.add_scalar(f"metric/val_time", validation_time, global_step=step)


        loss_values.update({"total": sum(loss_values.values())})
        pbar.update(loss_values)

        for key, value in loss_values.items():
            writer.add_scalar(f"loss/{key}", value, global_step=step)

        counter_profiler.new_group()

        if cfg.abort_step_size > 0 and step % cfg.abort_step_size == 0:
            logger.warning(f"Aborting training every {cfg.abort_step_size}")
            break

    time_profiler.end("training")

    if cfg.profile_time:
        result = time_profiler.get_results_string()
        logger.info(f"Time profiling:\n{result}")
        with open(out_root / "time.txt", "w", encoding="utf-8") as f:
            f.write(result + "\n")

    if cfg.profile_vram:
        write_json(out_root / "vram.json", vram_profiler.snapshots)

    if cfg.profile_counter:
        write_json(out_root / "counter.json", counter_profiler.data)

    writer.flush()


def find_latest_ckpt(folder: Path) -> Optional[Path]:
    if isfile(folder / "latest.ckpt"):
        return folder / "latest.ckpt"

    files = list(folder.glob("*.ckpt"))
    if len(files) == 0:
        return None

    files = sorted(
        [[int(file.stem), file] for file in files],
        key=lambda a: a[0]
    )
    return files[-1][1]


if __name__ == "__main__":
    main()

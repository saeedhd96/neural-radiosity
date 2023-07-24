import logging
import os
from pathlib import Path

import drjit as dr
import hydra
import mitsuba as mi
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from common import (compute_output_metrics, configure_compute, create_integrator, create_loss_function,
                    load_dataset, prepare_learned_objects)
from nerad.integrator.highquality import HighQuality
from nerad.model.config import ObjectConfig, TestConfig, TrainConfig
from nerad.utils.render_utils import render_and_save_image
from nerad.utils.sensor_utils import create_sensor
from nerad.utils.json_utils import write_json

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="test")
def main(cfg: TestConfig = None):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    use_hq = True

    device = os.environ.get("TORCH_DEVICE", "cuda:0")
    out_root = Path(HydraConfig.get().runtime.output_dir)  # experiment/test/ckpt

    # merge config from training
    train_root = out_root.parent.parent
    train_cfg = OmegaConf.load(train_root / ".hydra/config.yaml")
    cfg = merge_config(cfg, train_cfg)

    configure_compute(cfg.compute)

    scene, transforms, images, learned_modules = load_dataset(cfg.dataset, device)

    test_rendering = cfg.test_rendering
    test_integrators = {}
    for key, rendering in test_rendering.items():
        is_nerad = "nerad" in rendering.integrator
        integrator_injection = {}
        if is_nerad:
            residual_loss_function = create_loss_function(ObjectConfig("l2", {}), 0)  # dummy
            integrator_injection["residual_function"] = residual_loss_function
        integrator_function_injection = {"device": device}

        integrator = create_integrator(rendering, scene, cfg.dataset.scene, post_init_injection=integrator_injection,
                                       kwargs_injection=integrator_function_injection)

        test_integrators[key] = integrator
    logger.info(f"Integrators:\n{test_integrators}")

    # Load checkpoint
    learned_info = prepare_learned_objects(
        scene,
        test_integrators["image"],
        learned_modules,
        None,
        train_root / "checkpoints" / f"{cfg.ckpt}.ckpt",
        device,
    )

    if use_hq:
        block_size = cfg.blocksize
        logger.info(f"High quality renderer being used at block size: {block_size}")
        test_integrators = {
            k: HighQuality(block_size, v) for k, v in test_integrators.items()
        }

    view_indices = cfg.views
    if len(view_indices) == 0:
        n_views = 1 if cfg.n_views <= 0 else cfg.n_views
        view_indices = list(range(n_views))

    logger.info(f"Render {len(view_indices)} views to {out_root}")
    all_metrics = {}
    for idx in tqdm(view_indices):
        dr.flush_malloc_cache()
        torch.cuda.empty_cache()

        gt = {
            "image": images[idx] if images is not None else None,
        }
        view_metrics = {}

        for name, rendering in test_rendering.items():
            sensor = create_sensor(rendering.width, transforms[str(idx)])


            outputs = render_and_save_image(
                out_root / name,
                f"{idx:03d}",
                scene,
                test_integrators[name],
                rendering,
                sensor,
            )

            metrics = compute_output_metrics(name, outputs, rendering.integrator, gt)
            view_metrics.update(metrics)

        logger.info(f"Metrics for view {idx}\n" +
                    "\n".join((f"{k:>20} {v:.6f}" for k, v in view_metrics.items())) + "\n")
        all_metrics[str(idx)] = view_metrics

    write_json(out_root / "metrics.json", all_metrics)


def merge_config(test: TestConfig, train: TrainConfig) -> TestConfig:
    test.dataset = OmegaConf.merge(train.dataset, test.dataset) if test.dataset else train.dataset


    assert "image" in test.test_rendering, "Must have primary image integrator"

    cfg = test.test_rendering["image"]
    train_cfg = OmegaConf.to_container(train.rendering, resolve=True)
    cfg = OmegaConf.merge(train_cfg, cfg) if cfg else train_cfg

    test.test_rendering["image"] = cfg

    logger.info("Merged config:\n" + OmegaConf.to_yaml(test))
    return test


if __name__ == "__main__":
    main()

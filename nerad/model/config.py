from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ObjectConfig:
    name: str
    config: dict[str, Any]


@dataclass
class ComputeConfig:
    seed: int
    torch_detect_anomaly: bool
    dr_optimization_flags: bool


@dataclass
class TextureConfig:
    post_init: dict[str, Any]


@dataclass
class BsdfConfig:
    name: str
    post_init: dict[str, Any]
    texture: dict[str, TextureConfig]


@dataclass
class RenderingConfig:
    spp: int
    integrator: str
    config: dict[str, Any]
    post_init: dict[str, Any]
    width: int
    height: int


@dataclass
class DatasetConfig:
    scene: str
    cameras: Optional[str]
    n_views: int

@dataclass
class ValidationConfig:
    # unique name
    name: str
    # number of steps to do validation
    step_size: int
    # whether to validate after first step (sancheck)
    first_step: int

    rendering: RenderingConfig
    n_views: int


@dataclass
class SaveCheckpointConfig:
    step_size: int
    first_step: bool
    # if true, save to checkpoints/latest.ckpt
    # if False, save to checkpoints/[step].ckpt
    is_latest: bool




@dataclass
class TrainConfig:
    # saves output to out_root/out_dir
    # if out_dir is None, defaults to ${date}-${time}
    name: str
    out_root: str
    out_dir: Optional[str]
    # automatically resumes from latest checkpoint
    resume: bool
    validation: dict[str, ValidationConfig]
    saving: dict[str, SaveCheckpointConfig]

    rendering: RenderingConfig
    dataset: DatasetConfig
    batch_size: int

    learning_rate: float
    lr_decay_start: int
    lr_decay_rate: float
    lr_decay_steps: int
    lr_decay_min_rate: float

    beta_1: float
    beta_2: float

    residual_loss: Optional[ObjectConfig]

    n_steps: int
    shuffle: bool
    compute: ComputeConfig

    check_gradients: bool
    clipgrads: float
    profile_time: bool
    profile_time_sync_cuda: bool
    profile_vram: bool
    profile_counter: bool
    update_loss_step_size: int

    # watchdog use and debug
    is_watchdog_init: bool
    abort_step_size: int

    validation_view: int


@dataclass
class TestConfig:
    # checkpoint loaded from:
    # [experiment]/checkpoints/[ckpt].ckpt
    experiment: str
    ckpt: str

    compute: ComputeConfig
    test_rendering: dict[str, RenderingConfig]
    blocksize: int
    # below are from training by default
    # they can be overriden in command line
    dataset: DatasetConfig

    # if views are specified, n_views is ignored
    n_views: int
    views: list[int]




ConfigStore.instance().store("train_config", TrainConfig)
ConfigStore.instance().store("test_config", TestConfig)

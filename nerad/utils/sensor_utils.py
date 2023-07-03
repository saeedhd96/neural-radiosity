import random
import re
from pathlib import Path

import mitsuba as mi
import drjit as dr


def create_transforms(scene: str, n_views: int):
    # Hardcoded transformations only valid for lego scene

    path = str(Path(scene).parent / "cameras.xml")
    sensors = sensors = mi.load_file(path).sensors()
    transforms = {}
    for i in range(len(sensors)):
        fov = float(re.findall(r"\d+.*\d+", re.findall(r"x_fov = \[\d+.*\d+\]", str(sensors[i]))[0])[0])
        transforms[str(i)] = {
            "to_world": mi.ScalarTransform4f(sensors[i].world_transform().matrix.numpy()).matrix.numpy().tolist(),
            "fov": fov,
        }


    return transforms


def create_sensor(resolution, transform, random_crop=False, crop_size=None):
    return mi.load_dict(sensor_dict(resolution=resolution, fov=transform["fov"], to_world=transform["to_world"], random_crop=random_crop, crop_size=crop_size))


def sensor_dict(resolution, fov, to_world, random_crop=False, crop_size=None):
    sensor = {
        "type": "perspective",
        "fov": fov,
        "to_world": mi.ScalarTransform4f(to_world),
        "film": {
                "type": "hdrfilm",
                "width": resolution,
                "height": resolution,
                "filter": {"type": "box"},
                "pixel_format": "rgba"
        }
        # TODO: All scene MUST be rgba in this scenario and use a box filter, even for ground truth
    }

    if random_crop:
        assert crop_size > 0
        assert (resolution-crop_size) >= 0
        crop_offset = [
            random.randint(0, resolution-crop_size),
            random.randint(0, resolution-crop_size)
        ]
        sensor["film"]["crop_width"] = crop_size
        sensor["film"]["crop_height"] = crop_size
        sensor["film"]["crop_offset_x"] = crop_offset[0]
        sensor["film"]["crop_offset_y"] = crop_offset[1]

    return sensor

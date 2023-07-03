import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from mytorch.registry import Registry, import_children
from mytorch.utils.profiling_utils import counter_profiler, time_profiler
from nerad.utils.mitsuba_utils import vec_to_tens_safe


class MitsubaWrapper(nn.Module):
    def __init__(self, scene_min: float, scene_max: float, name: str = None):
        super().__init__()
        self.grad_activator = mi.Vector3f(0)
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.name = name or type(self).__name__

    def eval(self, pts, dirs=None, norms=None, albedo=None):
        if counter_profiler.enabled:
            counter_profiler.record(f"{self.name}.eval.pts", dr.shape(pts)[1])
        time_profiler.start(f"{self.name}.eval")
        pts = (pts - self.scene_min) / (self.scene_max - self.scene_min)
        result = self._eval(pts, dirs, norms, albedo)
        time_profiler.end(f"{self.name}.eval")
        return result

    def traverse(self, callback):
        callback.put_parameter("grad_activator", self.grad_activator, mi.ParamFlags.Differentiable)
        self._traverse(callback)

    def _eval(self, pts, dirs, norms, albedo):
        raise NotImplementedError()

    def _traverse(self, callback):
        pass



wrapper_registry = Registry("wrapper", MitsubaWrapper)
import_children(__file__, __name__)

import re
from pathlib import Path

import drjit as dr
import mitsuba as mi


def vec_to_tens_safe(vec):
    # Converts a Vector3f to a TensorXf safely in mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(dr.ravel(vec), shape=[dr.shape(vec)[1], dr.shape(vec)[0]])


def get_batch_size(spp):
    """
    Get the maximum power of 2 batch size possible given the 2^30 limit by mitsuba for the wavefront size
    """
    maximum_wavefrontsize= 2**30
    return 2**int(dr.log2(maximum_wavefrontsize/spp)/2)

import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.integrator import register_integrator

from .nerad import Nerad


@register_integrator("nerad_specular")
class NeradSpecular(Nerad, nn.Module):

    def is_specular(self, si):
            bsdf = si.bsdf()
            return mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta)

    def trace_speculars(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               active: mi.Bool,
               prev_si: mi.SurfaceInteraction3f,
               prev_bsdf_pdf: mi.Float,
               prev_bsdf_delta: mi.Bool,
               valid_ray: mi.Bool,
               throughput: mi.Spectrum,
               eta: mi.Float
               ):

        initial_si = None
        if isinstance(ray, mi.SurfaceInteraction3f):
            initial_si = ray
            ray = dr.zeros(mi.Ray3f)

        depth = mi.UInt32(0)

        active = mi.Bool(active)                      # Active SIMD lanes

        final_si = dr.zeros(mi.SurfaceInteraction3f)
        bsdf_ctx = mi.BSDFContext()

        # Record the following loop in its entirety
        loop = mi.Loop(name="MyWhittedRayTracer",
                       state=lambda: (sampler, ray, throughput,
                                      eta, depth, prev_si, prev_bsdf_pdf,
                                      prev_bsdf_delta, active))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            if initial_si is not None:
                si = initial_si
                initial_si = None
                bsdf = si.bsdf()
            else:
                si = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All,
                                        coherent=dr.eq(depth, 0), active = active)

                # Get the BSDF, potentially computes texture-space differentials
                bsdf = si.bsdf(ray)

            final_si = dr.select(active, si, final_si)
            speculars = self.is_specular(si)

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid() & speculars

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
                sampler, active, bsdf_ctx, si, bsdf, active_next)

            # ------ Update loop variables based on current interaction ------

            throughput *= dr.select(active_next , bsdf_weight, 1)
            eta *= dr.select(active_next , bsdf_sample.eta, 1)
            valid_ray |= active & si.is_valid() & ~mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Null)

            # Information about the current vertex needed by the next iteration
            prev_si = dr.select(active_next, si, prev_si)
            prev_bsdf_pdf = dr.select(active_next, bsdf_sample.pdf, prev_bsdf_pdf)
            prev_bsdf_delta = dr.select(active_next, mi.has_flag( bsdf_sample.sampled_type, mi.BSDFFlags.Delta), prev_bsdf_delta)

            # -------------------- Stopping criterion ---------------------

            #TODO: make sure this part is correct

            depth[si.is_valid()] += 1
            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            rr_prob = dr.minimum(throughput_max * eta**2, .95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob
            #throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)

        return final_si, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta


    def to_string(self):
        return (
            "NeradSpecularIntegrator[\n"
            f"  network={self.network}\n"
            f"  residual_function={self.residual_function}\n"
            f"  max_depth={self.max_depth}\n"
            "]"
        )

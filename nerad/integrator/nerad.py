import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn

from nerad.integrator import register_integrator
from nerad.loss import LossFucntion
from nerad.mitsuba_wrapper import wrapper_registry
from nerad.model.sampler import ShapeSampler
from nerad.texture.dictionary import MiDictionary

from .path import MyPathTracer


@register_integrator("nerad")
class Nerad(MyPathTracer, nn.Module):
    def __init__(self, props: mi.Properties):
        nn.Module.__init__(self)
        MyPathTracer.__init__(self, props)
        self.residual_sampler = None
        self.residual_sampler_m = None
        self.residual_function = None
        self.network = None
        self.return_only_LHS = props.get("config").dict.get("return_only_LHS")
        self.m = props.get("config").dict.get("m")

    def post_init(
        self,
        residual_function: LossFucntion,
        function: str,
        kwargs: MiDictionary,
    ):
        self.residual_function = residual_function
        self.network = wrapper_registry.build(function, kwargs)

    def get_albedo_detached(self, si):
        with dr.suspend_grad():
            with torch.no_grad():
                reflect = si.bsdf().eval_diffuse_reflectance(si)
        return reflect

    def compute_residual(self, scene, n, seed):
        if self.residual_sampler is None:
            self.residual_sampler = ShapeSampler(scene, no_specular_samples=True)
            assert self.m != 0
            self.residual_sampler_m = self.residual_sampler.sampler.clone()
            self.residual_sampler_m.seed(seed,n*self.m)

        self.residual_sampler_m.schedule_state()
        si, _ = self.residual_sampler.sample_input(scene=scene, n=n, seed=seed)
        _, _, aov = self.sample(scene, self.residual_sampler.sampler, si, 0, True, sampler_m = self.residual_sampler_m)
        residual = mi.Color3f(aov[-3:])
        return residual

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool,
               **kwargs):

        m = 1
        sampler_m = kwargs.get("sampler_m", None)
        if sampler_m is not None:
            m = self.m

        depth = mi.UInt32(0)
        eta = mi.Float(1)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None))

        active = mi.Bool(active)                      # Active SIMD lanes

        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        if isinstance(ray, mi.SurfaceInteraction3f):
            si = ray
            bsdf = si.bsdf()
            #Assertion: there shouldn't be ANY specualr sample here assuming that the code gets here only when residual sampling
            assert (dr.none(mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.Delta)))
        else:
            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))
            bsdf = si.bsdf(ray)

        # ---------------------- Handle specular surfaces (if any) ----------------------

        si, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta = self.trace_speculars(scene, sampler, si, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta)
        bsdf = si.bsdf()

        # ---------------------- Eval LHS ----------------------
        pts, dirs, normals, albedo = self.extract_inputs(si)
        LHS = self.network.eval(pts, dirs, normals, albedo)
        LHS = dr.select(active & si.is_valid(), throughput*LHS, mi.Vector3f(0))

        # ---------------------- Direct emission ----------------------

        E = self.emitter_hit(scene, throughput, prev_si,
                             prev_bsdf_pdf, prev_bsdf_delta, si)

        if self.return_only_LHS:
            mask = valid_ray | (active & si.is_valid())
            LHS = dr.select(mask, E + LHS, 0)
            zero_vec = LHS*0
            return zero_vec, mask, [LHS.x, LHS.y, LHS.z, dr.select(mask, mi.Float(1), mi.Float(0)), zero_vec.x, zero_vec.y, zero_vec.z]


        # ---------------------- repeat (if requested) ----------------------
        if m > 1:
            indices = dr.arange(mi.UInt, 0, len(si.p[0]))
            indices = dr.repeat(indices, self.m)
            si = dr.gather(type(si), si, indices)
            prev_bsdf_delta = dr.gather(type(prev_bsdf_delta), prev_bsdf_delta, indices)
            prev_bsdf_pdf = dr.gather(type(prev_bsdf_pdf), prev_bsdf_pdf, indices)
            prev_si = dr.gather(type(si), prev_si, indices)
            throughput = dr.gather(type(throughput), throughput, indices)
            eta = dr.gather(type(eta), eta, indices)
            valid_ray = dr.gather(type(valid_ray), valid_ray, indices)


            bsdf = si.bsdf()
            sampler = sampler_m

        # ---------------------- Emitter sampling ----------------------

        active_next = si.is_valid()

        em_sample_result = self.sample_emitter(
            scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)

        # ------------------ Detached BSDF sampling -------------------

        bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
            sampler, active, bsdf_ctx, si, bsdf, active_next)

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= active & si.is_valid() & ~mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Null)

        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1
        active = active_next

        si = scene.ray_intersect(ray,
                                 ray_flags=mi.RayFlags.All,
                                 coherent=dr.eq(depth, 0))

        bsdf = si.bsdf(ray)

        # ---------------------- Handle specular surfaces (if any) ----------------------

        si, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta = self.trace_speculars(scene, sampler, si, active, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta)
        bsdf = si.bsdf()

        # ---------------------- Direct emission ----------------------

        bsdf_sample_result = self.emitter_hit(
            scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)

        # ---------------------- Eval RHS ----------------------
        with dr.suspend_grad():
            with torch.no_grad():
                pts, dirs, normals, albedo = self.extract_inputs(si)
                RHS_net = dr.select(active & si.is_valid(),
                                    self.network.eval(pts, -ray.d, normals, albedo), mi.Vector3f(0))

        RHS = RHS_net * throughput + bsdf_sample_result + em_sample_result

        # ---------------------- Deal with repeat (if any) ----------------------
        if m > 1:
            RHS = dr.block_sum(RHS, self.m)/self.m
            validity = dr.select(valid_ray, mi.Float(1), mi.Float(0))
            valid_ray = dr.block_sum(validity, self.m)>0


        aov = dr.select(valid_ray, E + LHS, 0)
        rgb = dr.select(valid_ray, E + RHS, 0)

        residual = dr.select(valid_ray, self.residual_function.compute_loss(LHS, RHS), 0)


        return rgb, valid_ray, [aov.x, aov.y, aov.z, dr.select(valid_ray, mi.Float(1), mi.Float(0)), residual.x, residual.y, residual.z]

    def extract_inputs(self, si):
        pts = si.p
        dirs = si.to_world(si.wi)
        normals = si.sh_frame.n
        normals = dr.select(dr.dot(dirs, normals)<0, -normals, normals)
        albedo = dr.detach(self.get_albedo_detached(si))
        return pts,dirs,normals,albedo

    def aov_names(self):
        return ["LHS.R", "LHS.G", "LHS.B", "LHS.a", "residual.x", "residual.y", "residual.z"]

    def to_string(self):
        return (
            "NeradIntegrator[\n"
            f"  network={self.network}\n"
            f"  residual_function={self.residual_function}\n"
            "]"
        )

    def traverse(self, callback):
        self.network.traverse(callback)


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
                #Implemented in the child Nerad Specular
        return ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta, valid_ray, throughput, eta

    def is_specular(self, si):
            return si.is_valid() & False

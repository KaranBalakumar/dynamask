import torch
from types import SimpleNamespace
import time
import math
import pypose as pp
from contextlib import nullcontext

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Module.Map import VisualMap
from Utility.Timer import Timer
from Utility.Math  import NormalizeQuat

from ..Interface import IOptimizer
from ..PyposeOptimizers import LM_analytic, AnalyticModule, FactorGraph
from .Graphs import GraphInput, GraphOutput
from .Graphs import ICP_TwoframePGO, Reproj_TwoFramePGO, ReprojDisp_TwoFramePGO, Reproj_TwoFramePGO_IMU
from .Graphs import Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO


class TwoFrame_PGO(IOptimizer[GraphInput, dict, GraphOutput]):
    @staticmethod
    def _weighted_chi2(residual: torch.Tensor, cov_blocks: torch.Tensor | list[torch.Tensor]) -> float:
        r = residual.detach().double().reshape(-1)
        if isinstance(cov_blocks, list):
            off = 0
            val = 0.0
            for cov in cov_blocks:
                dim = int(cov.shape[0])
                rr = r[off : off + dim].unsqueeze(-1)
                inv = torch.pinverse(cov.double())
                val += float((rr.transpose(0, 1) @ inv @ rr).item())
                off += dim
            return val

        block = cov_blocks.double()
        n = int(block.shape[0])
        dim = int(block.shape[1])
        rr = r.view(n, dim)
        inv = torch.pinverse(block)
        maha = torch.einsum("ni,nij,nj->n", rr, inv, rr)
        return float(maha.sum().item())

    @staticmethod
    def _vis_from_residual(residual: torch.Tensor, prefer_dim: int = 2) -> torch.Tensor:
        flat = residual.detach().double().reshape(-1)
        if prefer_dim > 0 and flat.numel() % prefer_dim == 0:
            return flat.view(-1, prefer_dim)
        if flat.numel() % 3 == 0:
            return flat.view(-1, 3)
        return flat.unsqueeze(-1)

    @staticmethod
    def _percentile(values: torch.Tensor, q: float) -> float:
        if values.numel() == 0:
            return 0.0
        return float(torch.quantile(values.double(), q).item())

    @staticmethod
    def _safe_cond(mat: torch.Tensor) -> float:
        try:
            return float(torch.linalg.cond(mat.double()).item())
        except Exception:
            return float("nan")

    @staticmethod
    def _mahalanobis_per_obs(
        residual_rows: torch.Tensor,
        cov_blocks: torch.Tensor | list[torch.Tensor],
        imu_appended: bool,
    ) -> torch.Tensor:
        if residual_rows.numel() == 0:
            return torch.zeros((0,), dtype=torch.float64)
        if isinstance(cov_blocks, list):
            visual_covs = cov_blocks[:-1] if imu_appended and len(cov_blocks) > 0 else cov_blocks
            n = min(len(visual_covs), residual_rows.shape[0])
            out = torch.zeros((n,), dtype=torch.float64, device=residual_rows.device)
            for i in range(n):
                r = residual_rows[i].double().unsqueeze(-1)
                inv = torch.pinverse(visual_covs[i].double())
                out[i] = torch.sqrt(torch.clamp((r.transpose(0, 1) @ inv @ r).squeeze(), min=0.0))
            return out
        inv = torch.pinverse(cov_blocks.double())
        rr = residual_rows.double()
        maha_sq = torch.einsum("ni,nij,nj->n", rr, inv, rr)
        return torch.sqrt(torch.clamp(maha_sq, min=0.0))

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]
        from_candidates = global_map.match2frame1.project(obs.index)
        if from_candidates.numel() == 0:
            from_idx = frame_idx - 1
        else:
            from_idx = torch.unique(from_candidates)[:1]

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        init_motion = pp.SE3(frame2opt.data["pose"])
        baseline = frame2opt.data["baseline"]
        graph_data = GraphInput(frame_idx, from_idx, init_motion, baseline, obs, pts, im_intrinsics, edges_idx, "cpu")
        if self.config.graph_type == "reproj_imu":
            from_pose = global_map.frames.data["pose"][from_idx]
            imu_edge = global_map.get_imu_edge(from_idx, frame_idx)
            if imu_edge is None:
                raise RuntimeError(f"Missing IMU edge for frames ({int(from_idx.item())}, {int(frame_idx.item())})")
            graph_data.from_pose = from_pose
            graph_data.T_BS = frame2opt.data["T_BS"]
            graph_data.imu_edge = imu_edge
            graph_data.init_v_i = global_map.frames.data["vel"][from_idx].double().reshape(-1)
            graph_data.init_v_j = frame2opt.data["vel"][0].double()
            graph_data.init_bg_i = global_map.frames.data["bias_g"][from_idx].double().reshape(-1)
            graph_data.init_bg_j = frame2opt.data["bias_g"][0].double()
            graph_data.init_ba_i = global_map.frames.data["bias_a"][from_idx].double().reshape(-1)
            graph_data.init_ba_j = frame2opt.data["bias_a"][0].double()
            graph_data.gravity = getattr(global_map, "gravity", torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64))
        return graph_data

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"icp", "reproj", "disp", "reproj_imu"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool)
        })

    @staticmethod
    def init_context(config) -> dict:
        match (config.autodiff, config.graph_type):
            case (True, "icp"):
                PoseGraphClass = ICP_TwoframePGO
            case (True, "reproj"):
                PoseGraphClass = Reproj_TwoFramePGO
            case (True, "disp"):
                PoseGraphClass = ReprojDisp_TwoFramePGO
            case (True, "reproj_imu"):
                PoseGraphClass = Reproj_TwoFramePGO_IMU
            case (False, "icp"):
                PoseGraphClass = Analytic_ICP_TwoframePGO
            case (False, "reproj"):
                PoseGraphClass = Analytic_Reproj_TwoFramePGO
            case (False, "disp"):
                PoseGraphClass = Analytic_ReprojDisp_TwoFramePGO
            case (False, "reproj_imu"):
                PoseGraphClass = Reproj_TwoFramePGO_IMU
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        return {
            "optimizer_cfg": {
                "kernel"   : Huber(delta=0.1),
                "solver"   : PINV(),
                "strategy" : TrustRegion(radius=1e3),
                "corrector": FastTriggs(Huber(delta=0.1)),
                "vectorize": config.vectorize,
            },
            "device": config.device,

            "pose_graph_class": PoseGraphClass
        }

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        t0 = time.perf_counter()
        gpu_ctx = Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()) if torch.cuda.is_available() else nullcontext()
        with Timer.CPUTimingContext("TwoframePGO"), gpu_ctx:
            graph: FactorGraph = context["pose_graph_class"](graph_data)\
                .to(device=torch.device(context["device"]), dtype=torch.double)
            assert isinstance(graph, FactorGraph)
            residual_init = graph().detach().reshape(-1)
            cov_init = graph.covariance_array()

            if isinstance(graph, AnalyticModule):
                optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
            else:
                optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])

            scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)
            lm_iters = 0

            while scheduler.continual():
                cov_blocks = graph.covariance_array()
                if isinstance(cov_blocks, list):
                    inv_blocks = [torch.pinverse(c.to(context["device"]).double()) for c in cov_blocks]
                else:
                    inv_stacked = torch.pinverse(cov_blocks.to(context["device"]).double())
                    inv_blocks = [*torch.unbind(inv_stacked, dim=0)]
                weight = torch.block_diag(*inv_blocks)
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)
                lm_iters += 1
            residual_final = graph().detach().reshape(-1)
            cov_final = graph.covariance_array()

        out = graph.write_back()
        chi2_init = TwoFrame_PGO._weighted_chi2(residual_init, cov_init)
        chi2_final = TwoFrame_PGO._weighted_chi2(residual_final, cov_final)
        chi2_delta = (chi2_init - chi2_final) / max(chi2_init, 1e-12)
        vis_res_init = residual_init
        vis_res_final = residual_final
        imu_appended = isinstance(graph, Reproj_TwoFramePGO_IMU)
        if isinstance(graph, Reproj_TwoFramePGO_IMU):
            vis_len = int(graph.kp2.shape[0] * 2)
            vis_res_init = residual_init[:vis_len]
            vis_res_final = residual_final[:vis_len]
        if isinstance(cov_final, list):
            visual_covs = cov_final[:-1] if imu_appended and len(cov_final) > 0 else cov_final
            vis_dim = int(visual_covs[0].shape[0]) if len(visual_covs) > 0 else 1
            vis_before = vis_res_init.view(-1, vis_dim) if vis_res_init.numel() % max(vis_dim, 1) == 0 else vis_res_init.unsqueeze(-1)
            vis_after = vis_res_final.view(-1, vis_dim) if vis_res_final.numel() % max(vis_dim, 1) == 0 else vis_res_final.unsqueeze(-1)
        else:
            vis_dim = int(cov_final.shape[1]) if cov_final.ndim == 3 else 1
            vis_before = vis_res_init.view(-1, vis_dim) if vis_res_init.numel() % max(vis_dim, 1) == 0 else vis_res_init.unsqueeze(-1)
            vis_after = vis_res_final.view(-1, vis_dim) if vis_res_final.numel() % max(vis_dim, 1) == 0 else vis_res_final.unsqueeze(-1)
        vis_norm_after = torch.linalg.vector_norm(vis_after, dim=-1)
        vis_maha = TwoFrame_PGO._mahalanobis_per_obs(vis_after, cov_final, imu_appended=imu_appended)
        jac_cond = 0.0
        if isinstance(graph, AnalyticModule):
            jac_cond = TwoFrame_PGO._safe_cond(graph.jacobian())
            if not math.isfinite(jac_cond):
                jac_cond = 0.0

        imu_stats = {
            "backend.imu_residual.r_R.norm": 0.0,
            "backend.imu_residual.r_v.norm": 0.0,
            "backend.imu_residual.r_p.norm": 0.0,
            "backend.imu_residual.r_bg.norm": 0.0,
            "backend.imu_residual.r_ba.norm": 0.0,
            "backend.imu_residual.maha.total": 0.0,
        }
        r_imu_before = torch.zeros((15,), dtype=torch.float64)
        r_imu_after = torch.zeros((15,), dtype=torch.float64)
        if isinstance(graph, Reproj_TwoFramePGO_IMU):
            vis_len = int(graph.kp2.shape[0] * 2)
            r_imu_before = residual_init[vis_len : vis_len + 15]
            r_imu_after = residual_final[vis_len : vis_len + 15]
            imu_stats["backend.imu_residual.r_R.norm"] = float(torch.linalg.vector_norm(r_imu_after[0:3]).item())
            imu_stats["backend.imu_residual.r_v.norm"] = float(torch.linalg.vector_norm(r_imu_after[3:6]).item())
            imu_stats["backend.imu_residual.r_p.norm"] = float(torch.linalg.vector_norm(r_imu_after[6:9]).item())
            imu_stats["backend.imu_residual.r_bg.norm"] = float(torch.linalg.vector_norm(r_imu_after[9:12]).item())
            imu_stats["backend.imu_residual.r_ba.norm"] = float(torch.linalg.vector_norm(r_imu_after[12:15]).item())
            if isinstance(cov_final, list) and len(cov_final) > 0:
                inv_imu = torch.pinverse(cov_final[-1].double())
                imu_stats["backend.imu_residual.maha.total"] = float((r_imu_after.unsqueeze(0) @ inv_imu @ r_imu_after.unsqueeze(-1)).item())

        w_eps_hits = 0
        if isinstance(cov_final, list):
            for cov in cov_final:
                eig_min = float(torch.linalg.eigvalsh(cov.double()).min().item())
                if eig_min <= 1e-12:
                    w_eps_hits += 1
        else:
            eig_min = torch.linalg.eigvalsh(cov_final.double()).min(dim=-1).values
            w_eps_hits = int((eig_min <= 1e-12).sum().item())

        diagnostics = {
            "backend.twoframe.lm.iters": float(lm_iters),
            "backend.twoframe.lm.converged": float(1.0 if (lm_iters > 0 and chi2_delta > 1e-4 and math.isfinite(chi2_final)) else 0.0),
            "backend.twoframe.chi2.init": float(chi2_init),
            "backend.twoframe.chi2.final": float(chi2_final),
            "backend.twoframe.chi2.delta_frac": float(chi2_delta),
            "backend.twoframe.r_vis.norm.p50": TwoFrame_PGO._percentile(vis_norm_after, 0.5),
            "backend.twoframe.r_vis.norm.p95": TwoFrame_PGO._percentile(vis_norm_after, 0.95),
            "backend.twoframe.r_vis.maha.p50": TwoFrame_PGO._percentile(vis_maha, 0.5),
            "backend.twoframe.r_vis.maha.gt_3sig_frac": float((vis_maha > 3.0).float().mean().item()) if vis_maha.numel() > 0 else 0.0,
            "backend.twoframe.w_eps_hits": float(w_eps_hits),
            "backend.twoframe.jacobian.cond": float(jac_cond),
            "backend.twoframe.time_ms": float((time.perf_counter() - t0) * 1000.0),
            **imu_stats,
        }
        out.diagnostics = diagnostics
        out.dump_payload = {
            "stage": torch.tensor(0, dtype=torch.int64),
            "r_vis_before": vis_before.cpu(),
            "r_vis_after": vis_after.cpu(),
            "r_imu_before": r_imu_before.cpu(),
            "r_imu_after": r_imu_after.cpu(),
            "chi2_init": torch.tensor(chi2_init, dtype=torch.float64),
            "chi2_final": torch.tensor(chi2_final, dtype=torch.float64),
            "lm_iters": torch.tensor(lm_iters, dtype=torch.int64),
        }
        return context, out

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        step = int(result.frame_idx.item())
        logger = getattr(self, "debug_logger", None)
        old_pose = pp.SE3(global_map.frames.data["pose"][result.frame_idx].double())
        old_vel = global_map.frames.data["vel"][result.frame_idx].double()
        old_bg = global_map.frames.data["bias_g"][result.frame_idx].double()
        old_ba = global_map.frames.data["bias_a"][result.frame_idx].double()

        to_pose     = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
        if result.vel is not None:
            global_map.frames.data["vel"][result.frame_idx] = result.vel.float().cpu()
        if result.bias_g is not None:
            global_map.frames.data["bias_g"][result.frame_idx] = result.bias_g.float().cpu()
        if result.bias_a is not None:
            global_map.frames.data["bias_a"][result.frame_idx] = result.bias_a.float().cpu()
        new_pose = pp.SE3(global_map.frames.data["pose"][result.frame_idx].double())
        new_vel = global_map.frames.data["vel"][result.frame_idx].double()
        new_bg = global_map.frames.data["bias_g"][result.frame_idx].double()
        new_ba = global_map.frames.data["bias_a"][result.frame_idx].double()

        trans_delta = float(torch.linalg.vector_norm(new_pose.translation() - old_pose.translation()).item())
        R_delta = old_pose.rotation().matrix().transpose(-1, -2) @ new_pose.rotation().matrix()
        tr = float(torch.trace(R_delta[0]).item())
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5))))
        writeback = {
            "backend.writeback.pose.delta.trans_m": trans_delta,
            "backend.writeback.pose.delta.rot_deg": float(angle),
            "backend.writeback.vel.delta_norm": float(torch.linalg.vector_norm(new_vel - old_vel).item()),
            "backend.writeback.bias_g.delta_norm": float(torch.linalg.vector_norm(new_bg - old_bg).item()),
            "backend.writeback.bias_a.delta_norm": float(torch.linalg.vector_norm(new_ba - old_ba).item()),
            "backend.writeback.bias_g.norm": float(torch.linalg.vector_norm(new_bg).item()),
            "backend.writeback.bias_a.norm": float(torch.linalg.vector_norm(new_ba).item()),
            "diag.backend.writeback.pose_jump_warn": float(1.0 if (trans_delta > 1.0 or angle > 10.0) else 0.0),
        }
        if logger is not None:
            if result.diagnostics is not None:
                logger.log_scalars(result.diagnostics, step)
            logger.log_scalars(writeback, step)
            if result.dump_payload is not None:
                logger.dump_artifact("pgo", {"pgo": result.dump_payload}, step)


class Local_TwoFrame_PGO(TwoFrame_PGO):
    """
    Simple two-frame PGO in visual-odometry (MAC-VO) under Local frame. May lead to better optimization
    due to more numerical stability (especially in large-scene with 1000+ meters size)
    """
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        global_graph_data = super().get_graph_data(global_map, frame_idx, observations, edges)
        self.T_o2w_idx = frame_idx - 1

        T_o2w = pp.SE3(global_map.frames.data["pose"][frame_idx - 1])
        T_w2o = T_o2w.Inv()
        return self.world_to_optim(global_graph_data, T_w2o)

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return

        T_o2w = pp.SE3(global_map.frames.data["pose"][self.T_o2w_idx])
        super().write_graph_data(self.optim_to_world(result, T_o2w), global_map)

    def world_to_optim(self, data: GraphInput, T_w2o: pp.LieTensor) -> GraphInput:
        """Transform the optimization graph data into local reference frame (i.e. the reference frame is the pose of previous key frame)
        """
        # Same for below:
        # c = camera to optimize, o = optimization frame, w = world (global) frame
        T_c2w = pp.LieTensor(data.init_motion, ltype=pp.SE3_type)
        T_c2o = T_w2o @ T_c2w
        R_w2o = T_w2o.rotation().matrix().to(data.points.data["cov_Tw"])

        data.init_motion = T_c2o
        data.points.data["pos_Tw"]  = pp.Act(pp.SE3(T_w2o.to(data.points.data["pos_Tw"])), data.points.data["pos_Tw"])
        data.points.data["cov_Tw"]  = R_w2o @ data.points.data["cov_Tw"] @ R_w2o.transpose(-1, -2)
        return data

    def optim_to_world(self, data: GraphOutput, T_o2w: pp.LieTensor) -> GraphOutput:
        """Transform the optimization result under local reference frame (w.r.t. previous KF) to the global frame.
        """
        T_c2o = data.motion
        data.motion = NormalizeQuat(T_o2w @ pp.SE3(T_c2o.to(T_o2w)))
        return data


class Empty_TwoFrame_PGO(TwoFrame_PGO):
    """
    A 'no-op' variant of the Two-frame PGO optimizer. Helpful in debugging process.
    """
    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        return context, GraphOutput(motion=graph_data.init_motion,
                                    frame_idx=graph_data.frame_idx,
                                    from_idx=graph_data.from_idx)

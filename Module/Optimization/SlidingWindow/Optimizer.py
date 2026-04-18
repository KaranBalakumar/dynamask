from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import math
import time
from contextlib import nullcontext

import torch
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Module.Map import VisualMap
from Utility.Timer import Timer

from ..Interface import IOptimizer
from ..PyposeOptimizers import LM_analytic, AnalyticModule, FactorGraph
from ..TwoFramePGO.Graphs import (
    GraphInput as PairGraphInput,
    GraphOutput as PairGraphOutput,
    ICP_TwoframePGO,
    Reproj_TwoFramePGO,
    ReprojDisp_TwoFramePGO,
    Reproj_TwoFramePGO_IMU,
    Analytic_ICP_TwoframePGO,
    Analytic_Reproj_TwoFramePGO,
    Analytic_ReprojDisp_TwoFramePGO,
)


@dataclass
class SWGraphInput:
    pairs: list[PairGraphInput]


@dataclass
class SWGraphOutput:
    results: list[PairGraphOutput]
    diagnostics: dict[str, float] | None = None
    dump_payload: dict[str, torch.Tensor] | None = None


class SlidingWindow_VIO_PGO(IOptimizer[SWGraphInput, dict, SWGraphOutput]):
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"icp", "reproj", "disp", "reproj_imu"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool),
            "window_size": lambda n: isinstance(n, int) and n >= 2,
        }, allow_excessive_cfg=True)

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
                raise ValueError(f"Unsupported graph type: {config.graph_type}")

        return {
            "optimizer_cfg": {
                "kernel": Huber(delta=0.1),
                "solver": PINV(),
                "strategy": TrustRegion(radius=1e3),
                "corrector": FastTriggs(Huber(delta=0.1)),
                "vectorize": config.vectorize,
            },
            "device": config.device,
            "pose_graph_class": PoseGraphClass,
        }

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor, observations=None, edges=None) -> SWGraphInput:
        end = int(frame_idx.item())
        win = int(getattr(self.config, "window_size", 5))
        start = max(1, end - win + 1)
        pairs: list[PairGraphInput] = []

        for cur in range(start, end + 1):
            cur_idx = torch.tensor([cur], dtype=torch.long)
            frame2opt = global_map.frames[cur_idx]
            obs = global_map.get_frame2match(frame2opt)
            im_intrinsics = frame2opt.data["K"][0]
            from_all = global_map.match2frame1.mapping[obs.index]
            if self.config.graph_type == "reproj_imu":
                from_idx = cur_idx - 1
            elif from_all.numel() == 0:
                from_idx = cur_idx - 1
            else:
                from_valid = from_all[from_all >= 0]
                from_idx = (torch.unique(from_valid)[:1]) if from_valid.numel() > 0 else (cur_idx - 1)

            to_all = global_map.match2frame2.mapping[obs.index]
            pair_mask = (from_all == int(from_idx.item())) & (to_all == int(cur_idx.item()))
            if pair_mask.numel() == 0 or not bool(pair_mask.any()):
                continue
            obs = obs[pair_mask]
            pts = global_map.get_match2point(obs)
            edges_idx = torch.zeros((len(obs),), dtype=torch.long)
            init_motion = pp.SE3(frame2opt.data["pose"])
            baseline = frame2opt.data["baseline"]

            g = PairGraphInput(cur_idx, from_idx, init_motion, baseline, obs, pts, im_intrinsics, edges_idx, "cpu")
            if self.config.graph_type == "reproj_imu":
                imu_edge = global_map.get_imu_edge(from_idx, cur_idx)
                if imu_edge is None:
                    continue
                g.from_pose = global_map.frames.data["pose"][from_idx]
                g.T_BS = frame2opt.data["T_BS"]
                g.imu_edge = imu_edge
                g.init_v_i = global_map.frames.data["vel"][from_idx].double().reshape(-1)
                g.init_v_j = frame2opt.data["vel"][0].double()
                g.init_bg_i = global_map.frames.data["bias_g"][from_idx].double().reshape(-1)
                g.init_bg_j = frame2opt.data["bias_g"][0].double()
                g.init_ba_i = global_map.frames.data["bias_a"][from_idx].double().reshape(-1)
                g.init_ba_j = frame2opt.data["bias_a"][0].double()
                g.gravity = getattr(global_map, "gravity", torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64))
            pairs.append(g)

        return SWGraphInput(pairs=pairs)

    @staticmethod
    def _optimize(context: dict, graph_data: SWGraphInput) -> tuple[dict, SWGraphOutput]:
        results: list[PairGraphOutput] = []
        t0 = time.perf_counter()
        chi2_init_sum = 0.0
        chi2_final_sum = 0.0
        total_iters = 0.0
        n_imu_factors = 0
        gpu_ctx = Timer.GPUTimingContext("SlidingWindowPGO", torch.cuda.current_stream()) if torch.cuda.is_available() else nullcontext()
        with Timer.CPUTimingContext("SlidingWindowPGO"), gpu_ctx:
            for pair in graph_data.pairs:
                graph: FactorGraph = context["pose_graph_class"](pair).to(device=torch.device(context["device"]), dtype=torch.double)
                res_init = graph().detach().reshape(-1)
                cov_init = graph.covariance_array()
                if isinstance(graph, AnalyticModule):
                    optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
                else:
                    optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])
                scheduler = StopOnPlateau(optimizer, steps=8, patience=2, decreasing=1e-5, verbose=False)
                lm_iters = 0

                while scheduler.continual():
                    cov_blocks = graph.covariance_array()
                    if isinstance(cov_blocks, list):
                        inv_blocks = [torch.pinverse(c.to(context["device"]).double()) for c in cov_blocks]
                    else:
                        inv = torch.pinverse(cov_blocks.to(context["device"]).double())
                        inv_blocks = [*torch.unbind(inv, dim=0)]
                    weight = torch.block_diag(*inv_blocks)
                    loss = optimizer.step(input=(), weight=weight)
                    scheduler.step(loss)
                    lm_iters += 1

                res_final = graph().detach().reshape(-1)
                cov_final = graph.covariance_array()
                if isinstance(cov_init, list):
                    n_imu_factors += 1 if len(cov_init) > 0 else 0
                    off = 0
                    pair_init = 0.0
                    pair_final = 0.0
                    for c0, c1 in zip(cov_init, cov_final if isinstance(cov_final, list) else cov_init):
                        dim = int(c0.shape[0])
                        r0 = res_init[off : off + dim].unsqueeze(-1)
                        r1 = res_final[off : off + dim].unsqueeze(-1)
                        pair_init += float((r0.transpose(0, 1) @ torch.pinverse(c0.double()) @ r0).item())
                        pair_final += float((r1.transpose(0, 1) @ torch.pinverse(c1.double()) @ r1).item())
                        off += dim
                else:
                    inv0 = torch.pinverse(cov_init.double())
                    inv1 = torch.pinverse(cov_final.double())
                    dim = int(cov_init.shape[1])
                    r0 = res_init.view(-1, dim)
                    r1 = res_final.view(-1, dim)
                    pair_init = float(torch.einsum("ni,nij,nj->n", r0, inv0, r0).sum().item())
                    pair_final = float(torch.einsum("ni,nij,nj->n", r1, inv1, r1).sum().item())
                chi2_init_sum += pair_init
                chi2_final_sum += pair_final
                total_iters += float(lm_iters)
                results.append(graph.write_back())
        diagnostics = {
            "backend.swf.W_kf": float(len(graph_data.pairs)),
            "backend.swf.n_visual_factors": float(len(graph_data.pairs)),
            "backend.swf.n_imu_factors": float(n_imu_factors),
            "backend.swf.lm.iters": float(total_iters),
            "backend.swf.chi2.init": float(chi2_init_sum),
            "backend.swf.chi2.final": float(chi2_final_sum),
            "backend.swf.chi2.delta_frac": float((chi2_init_sum - chi2_final_sum) / max(chi2_init_sum, 1e-12)),
            "backend.swf.time_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        dump_payload = {
            "W_kf": torch.tensor(len(graph_data.pairs), dtype=torch.int64),
            "chi2_init": torch.tensor(chi2_init_sum, dtype=torch.float64),
            "chi2_final": torch.tensor(chi2_final_sum, dtype=torch.float64),
            "lm_iters": torch.tensor(total_iters, dtype=torch.float64),
        }
        return context, SWGraphOutput(results=results, diagnostics=diagnostics, dump_payload=dump_payload)

    def write_graph_data(self, result: SWGraphOutput | None, global_map: VisualMap) -> None:
        if result is None:
            return
        logger = getattr(self, "debug_logger", None)
        if logger is not None and result.diagnostics is not None and len(result.results) > 0:
            logger.log_scalars(result.diagnostics, int(result.results[-1].frame_idx.item()))
            if result.dump_payload is not None:
                logger.dump_artifact("pgo", {"swf": result.dump_payload}, int(result.results[-1].frame_idx.item()))
        for r in result.results:
            old_pose = pp.SE3(global_map.frames.data["pose"][r.frame_idx].double())
            old_vel = global_map.frames.data["vel"][r.frame_idx].double()
            old_bg = global_map.frames.data["bias_g"][r.frame_idx].double()
            old_ba = global_map.frames.data["bias_a"][r.frame_idx].double()
            to_pose = pp.SE3(r.motion[0].data.double().cpu())
            global_map.frames.data["pose"][r.frame_idx] = to_pose.float()
            if r.vel is not None:
                global_map.frames.data["vel"][r.frame_idx] = r.vel.float().cpu()
            if r.bias_g is not None:
                global_map.frames.data["bias_g"][r.frame_idx] = r.bias_g.float().cpu()
            if r.bias_a is not None:
                global_map.frames.data["bias_a"][r.frame_idx] = r.bias_a.float().cpu()
            if logger is not None:
                new_pose = pp.SE3(global_map.frames.data["pose"][r.frame_idx].double())
                new_vel = global_map.frames.data["vel"][r.frame_idx].double()
                new_bg = global_map.frames.data["bias_g"][r.frame_idx].double()
                new_ba = global_map.frames.data["bias_a"][r.frame_idx].double()
                trans_delta = float(torch.linalg.vector_norm(new_pose.translation() - old_pose.translation()).item())
                R_delta = old_pose.rotation().matrix().transpose(-1, -2) @ new_pose.rotation().matrix()
                tr = float(torch.trace(R_delta[0]).item())
                angle = math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5))))
                logger.log_scalars(
                    {
                        "backend.writeback.pose.delta.trans_m": trans_delta,
                        "backend.writeback.pose.delta.rot_deg": float(angle),
                        "backend.writeback.vel.delta_norm": float(torch.linalg.vector_norm(new_vel - old_vel).item()),
                        "backend.writeback.bias_g.delta_norm": float(torch.linalg.vector_norm(new_bg - old_bg).item()),
                        "backend.writeback.bias_a.delta_norm": float(torch.linalg.vector_norm(new_ba - old_ba).item()),
                        "backend.writeback.bias_g.norm": float(torch.linalg.vector_norm(new_bg).item()),
                        "backend.writeback.bias_a.norm": float(torch.linalg.vector_norm(new_ba).item()),
                        "diag.backend.writeback.pose_jump_warn": float(1.0 if (trans_delta > 1.0 or angle > 10.0) else 0.0),
                    },
                    int(r.frame_idx.item()),
                )

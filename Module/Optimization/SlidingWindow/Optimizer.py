from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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
            pts = global_map.get_match2point(obs)
            im_intrinsics = frame2opt.data["K"][0]
            from_candidates = global_map.match2frame1.project(obs.index)
            if from_candidates.numel() == 0:
                from_idx = cur_idx - 1
            else:
                from_idx = torch.unique(from_candidates)[:1]

            lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
            lengths = lengths[lengths >= 0]
            edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
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
        with Timer.CPUTimingContext("SlidingWindowPGO"), Timer.GPUTimingContext("SlidingWindowPGO", torch.cuda.current_stream()):
            for pair in graph_data.pairs:
                graph: FactorGraph = context["pose_graph_class"](pair).to(device=torch.device(context["device"]), dtype=torch.double)
                if isinstance(graph, AnalyticModule):
                    optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
                else:
                    optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])
                scheduler = StopOnPlateau(optimizer, steps=8, patience=2, decreasing=1e-5, verbose=False)

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

                results.append(graph.write_back())
        return context, SWGraphOutput(results=results)

    def write_graph_data(self, result: SWGraphOutput | None, global_map: VisualMap) -> None:
        if result is None:
            return
        for r in result.results:
            to_pose = pp.SE3(r.motion[0].data.double().cpu())
            global_map.frames.data["pose"][r.frame_idx] = to_pose.float()
            if r.vel is not None:
                global_map.frames.data["vel"][r.frame_idx] = r.vel.float().cpu()
            if r.bias_g is not None:
                global_map.frames.data["bias_g"][r.frame_idx] = r.bias_g.float().cpu()
            if r.bias_a is not None:
                global_map.frames.data["bias_a"][r.frame_idx] = r.bias_a.float().cpu()

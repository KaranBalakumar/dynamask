import torch
from types import SimpleNamespace
import pypose as pp

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
from .Graphs import ICP_TwoframePGO, Reproj_TwoFramePGO, ReprojDisp_TwoFramePGO, ReprojIMU_TwoFramePGO
from .Graphs import Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO


class TwoFrame_PGO(IOptimizer[GraphInput, dict, GraphOutput]):
    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None,
                       dynamic_gating_enabled: bool = False) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]
        frame_prev = global_map.frames[frame_idx - 1]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]
        imu = global_map.get_frame2imu(frame2opt)
        if len(imu) == 0:
            imu = None

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        init_motion = pp.SE3(frame2opt.data["pose"])
        baseline = frame2opt.data["baseline"]
        return GraphInput(
            frame_idx, frame_idx - 1, init_motion, baseline, obs, pts, im_intrinsics, edges_idx, "cpu",
            dynamic_gating_enabled=dynamic_gating_enabled, frame_from=frame_prev, frame_to=frame2opt, imu_factors=imu
        )

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
                PoseGraphClass = ReprojIMU_TwoFramePGO
            case (False, "icp"):
                PoseGraphClass = Analytic_ICP_TwoframePGO
            case (False, "reproj"):
                PoseGraphClass = Analytic_Reproj_TwoFramePGO
            case (False, "disp"):
                PoseGraphClass = Analytic_ReprojDisp_TwoFramePGO
            case (False, "reproj_imu"):
                raise ValueError("graph_type='reproj_imu' currently requires autodiff=true")
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
        with Timer.CPUTimingContext("TwoframePGO"), Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()):
            graph: FactorGraph = context["pose_graph_class"](graph_data)\
                .to(device=torch.device(context["device"]), dtype=torch.double)
            assert isinstance(graph, FactorGraph)

            if isinstance(graph, AnalyticModule):
                optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
            else:
                optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])

            scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)

            while scheduler.continual():
                cov_blocks = graph.covariance_array()
                if isinstance(cov_blocks, torch.Tensor):
                    inv_blocks = [torch.pinverse(c.to(context["device"]).double()) for c in cov_blocks]
                else:
                    inv_blocks = [torch.pinverse(c.to(context["device"]).double()) for c in cov_blocks]
                weight = torch.block_diag(*inv_blocks)
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)

        return context, graph.write_back()

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        
        to_pose     = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
        if result.vel_w is not None:
            global_map.frames.data["vel_w"][result.frame_idx] = result.vel_w[0].float().cpu()
        if result.bias_g is not None:
            global_map.frames.data["bias_g"][result.frame_idx] = result.bias_g[0].float().cpu()
        if result.bias_a is not None:
            global_map.frames.data["bias_a"][result.frame_idx] = result.bias_a[0].float().cpu()


class Local_TwoFrame_PGO(TwoFrame_PGO):
    """
    Simple two-frame PGO in visual-odometry (MAC-VO) under Local frame. May lead to better optimization
    due to more numerical stability (especially in large-scene with 1000+ meters size)
    """
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None,
                       dynamic_gating_enabled: bool = False) -> GraphInput:
        global_graph_data = super().get_graph_data(global_map, frame_idx, observations, edges, dynamic_gating_enabled)
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
        if data.frame_from is not None:
            from_pose = pp.SE3(data.frame_from.data["pose"])
            data.frame_from.data["pose"] = (T_w2o @ from_pose).float()
            data.frame_from.data["vel_w"] = (R_w2o @ data.frame_from.data["vel_w"].unsqueeze(-1)).squeeze(-1).float()
        if data.frame_to is not None:
            to_pose = pp.SE3(data.frame_to.data["pose"])
            data.frame_to.data["pose"] = (T_w2o @ to_pose).float()
            data.frame_to.data["vel_w"] = (R_w2o @ data.frame_to.data["vel_w"].unsqueeze(-1)).squeeze(-1).float()
        if data.imu_factors is not None:
            data.imu_factors.data["gravity_w"] = (R_w2o @ data.imu_factors.data["gravity_w"].unsqueeze(-1)).squeeze(-1).float()
        return data

    def optim_to_world(self, data: GraphOutput, T_o2w: pp.LieTensor) -> GraphOutput:
        """Transform the optimization result under local reference frame (w.r.t. previous KF) to the global frame.
        """
        T_c2o = data.motion
        data.motion = NormalizeQuat(T_o2w @ pp.SE3(T_c2o.to(T_o2w)))
        if data.vel_w is not None:
            R_o2w = T_o2w.rotation().matrix().to(data.vel_w)
            data.vel_w = (R_o2w @ data.vel_w.unsqueeze(-1)).squeeze(-1)
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

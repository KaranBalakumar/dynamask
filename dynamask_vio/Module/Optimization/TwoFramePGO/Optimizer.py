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
from Utility.Device import canonicalize_torch_device, is_supported_device_string

from ..Interface import IOptimizer
from ..PyposeOptimizers import LM_analytic, AnalyticModule, FactorGraph
from .Graphs import GraphInput, GraphOutput
from .Graphs import ICP_TwoframePGO, Reproj_TwoFramePGO, ReprojDisp_TwoFramePGO, SlidingWindow_ReprojDisp_PGO
from .Graphs import Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO


class TwoFrame_PGO(IOptimizer[GraphInput, dict, GraphOutput]):
    @staticmethod
    def _stack_window_imu_factors(global_map: VisualMap, frame_indices: torch.Tensor) -> dict[str, torch.Tensor] | None:
        factors: list[dict[str, torch.Tensor]] = []
        for i in range(frame_indices.numel() - 1):
            imu_factor = global_map.get_imu_factor(frame_indices[i:i + 1], frame_indices[i + 1:i + 2])
            if imu_factor is None:
                return None
            factors.append(imu_factor)
        if len(factors) == 0:
            return None
        keys = factors[0].keys()
        return {k: torch.cat([f[k] for f in factors], dim=0) for k in keys}

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        frame2opt = global_map.frames[frame_idx]
        from_pose = pp.SE3(global_map.frames.data["pose"][frame_idx - 1])
        init_vel_w = frame2opt.data["vel_w"]
        init_bias_g = frame2opt.data["bias_g"]
        init_bias_a = frame2opt.data["bias_a"]
        from_vel_w = global_map.frames.data["vel_w"][frame_idx - 1:frame_idx]
        from_bias_g = global_map.frames.data["bias_g"][frame_idx - 1:frame_idx]
        from_bias_a = global_map.frames.data["bias_a"][frame_idx - 1:frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        init_motion = pp.SE3(frame2opt.data["pose"])
        baseline = frame2opt.data["baseline"]
        imu_factor = global_map.get_imu_factor(frame_idx - 1, frame_idx)
        window_size = int(getattr(self.config, "window_size", 2))
        frame_scalar = int(frame_idx.reshape(-1)[0].item())
        start_idx = max(0, frame_scalar - window_size + 1)
        window_frame_idx = torch.arange(start_idx, frame_scalar + 1, dtype=torch.long)
        window_init_motion = pp.SE3(global_map.frames.data["pose"][window_frame_idx])
        window_init_vel_w = global_map.frames.data["vel_w"][window_frame_idx]
        window_init_bias_g = global_map.frames.data["bias_g"][window_frame_idx]
        window_init_bias_a = global_map.frames.data["bias_a"][window_frame_idx]
        window_imu_factors = self._stack_window_imu_factors(global_map, window_frame_idx) if window_frame_idx.numel() > 1 else None
        return GraphInput(
            frame_idx=frame_idx,
            from_idx=frame_idx - 1,
            init_motion=init_motion,
            from_pose=from_pose,
            baseline=baseline,
            observations=obs,
            points=pts,
            images_intrinsic=im_intrinsics,
            edges_index=edges_idx,
            device="cpu",
            imu_factor=imu_factor,
            init_vel_w=init_vel_w,
            init_bias_g=init_bias_g,
            init_bias_a=init_bias_a,
            from_vel_w=from_vel_w,
            from_bias_g=from_bias_g,
            from_bias_a=from_bias_a,
            window_frame_idx=window_frame_idx,
            window_init_motion=window_init_motion,
            window_init_vel_w=window_init_vel_w,
            window_init_bias_g=window_init_bias_g,
            window_init_bias_a=window_init_bias_a,
            window_imu_factors=window_imu_factors,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"icp", "reproj", "disp"},
            "device": is_supported_device_string,
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool),
        })
        if hasattr(config, "window_size"):
            cls._enforce_config_spec(getattr(config, "window_size"), lambda b: isinstance(b, int) and b >= 2)

    @staticmethod
    def init_context(config) -> dict:
        runtime_device = canonicalize_torch_device(config.device)
        window_size = int(getattr(config, "window_size", 2))
        if window_size > 2 and config.graph_type == "disp":
            PoseGraphClass = SlidingWindow_ReprojDisp_PGO
        else:
            match (config.autodiff, config.graph_type):
                case (True, "icp"):
                    PoseGraphClass = ICP_TwoframePGO
                case (True, "reproj"):
                    PoseGraphClass = Reproj_TwoFramePGO
                case (True, "disp"):
                    PoseGraphClass = ReprojDisp_TwoFramePGO
                case (False, "icp"):
                    PoseGraphClass = Analytic_ICP_TwoframePGO
                case (False, "reproj"):
                    PoseGraphClass = Analytic_Reproj_TwoFramePGO
                case (False, "disp"):
                    PoseGraphClass = Analytic_ReprojDisp_TwoFramePGO
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
            "device": runtime_device,
            "window_size": window_size,
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
                weight = torch.block_diag(*(
                    torch.pinverse(graph.covariance_array().to(context["device"]).double())
                ))
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)

        return context, graph.write_back()

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        
        to_pose = pp.SE3(result.motion[0].data.double().cpu())
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
        if result.vel_w is not None:
            global_map.frames.data["vel_w"][result.frame_idx] = result.vel_w[0].detach().cpu().float()
        if result.bias_g is not None:
            global_map.frames.data["bias_g"][result.frame_idx] = result.bias_g[0].detach().cpu().float()
        if result.bias_a is not None:
            global_map.frames.data["bias_a"][result.frame_idx] = result.bias_a[0].detach().cpu().float()
        if result.window_frame_idx is not None and result.window_motion is not None:
            global_map.frames.data["pose"][result.window_frame_idx] = pp.SE3(result.window_motion).tensor().detach().cpu().float()
            if result.window_vel_w is not None:
                global_map.frames.data["vel_w"][result.window_frame_idx] = result.window_vel_w.detach().cpu().float()
            if result.window_bias_g is not None:
                global_map.frames.data["bias_g"][result.window_frame_idx] = result.window_bias_g.detach().cpu().float()
            if result.window_bias_a is not None:
                global_map.frames.data["bias_a"][result.window_frame_idx] = result.window_bias_a.detach().cpu().float()


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
        if data.from_pose is not None:
            data.from_pose = T_w2o @ pp.SE3(data.from_pose)
        R_w2o = T_w2o.rotation().matrix().to(data.points.data["cov_Tw"])

        data.init_motion = T_c2o
        if data.init_vel_w is not None:
            data.init_vel_w = torch.bmm(R_w2o, data.init_vel_w.unsqueeze(-1)).squeeze(-1)
        if data.from_vel_w is not None:
            data.from_vel_w = torch.bmm(R_w2o, data.from_vel_w.unsqueeze(-1)).squeeze(-1)
        if data.window_init_motion is not None:
            data.window_init_motion = T_w2o @ pp.SE3(data.window_init_motion)
        if data.window_init_vel_w is not None:
            n = data.window_init_vel_w.shape[0]
            R_rep = R_w2o.repeat(n, 1, 1)
            data.window_init_vel_w = torch.bmm(R_rep, data.window_init_vel_w.unsqueeze(-1)).squeeze(-1)
        data.points.data["pos_Tw"]  = pp.Act(pp.SE3(T_w2o.to(data.points.data["pos_Tw"])), data.points.data["pos_Tw"])
        data.points.data["cov_Tw"]  = R_w2o @ data.points.data["cov_Tw"] @ R_w2o.transpose(-1, -2)
        return data

    def optim_to_world(self, data: GraphOutput, T_o2w: pp.LieTensor) -> GraphOutput:
        """Transform the optimization result under local reference frame (w.r.t. previous KF) to the global frame.
        """
        T_c2o = data.motion
        data.motion = NormalizeQuat(T_o2w @ pp.SE3(T_c2o.to(T_o2w)))
        R_o2w = T_o2w.rotation().matrix()
        if data.vel_w is not None:
            data.vel_w = torch.bmm(R_o2w, data.vel_w.unsqueeze(-1)).squeeze(-1)
        if data.window_motion is not None:
            data.window_motion = NormalizeQuat(T_o2w @ pp.SE3(data.window_motion.to(T_o2w)))
        if data.window_vel_w is not None:
            n = data.window_vel_w.shape[0]
            R_rep = R_o2w.repeat(n, 1, 1)
            data.window_vel_w = torch.bmm(R_rep, data.window_vel_w.unsqueeze(-1)).squeeze(-1)
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

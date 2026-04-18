import torch
import pypose as pp
import typing as T
from types import SimpleNamespace

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import StereoFrame, IMUData, StereoInertialFrame
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode, IMUEdgeNode
from Module.Network.AirIMU.contracts import narrow_corrector_output
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Timer import Timer
from Utility.Visualize import fig_plt
from Utility.Extensions import ConfigTestable
from Utility.Observability import DebugLogger, collect_runtime_dump, should_dump, cadence_reason
from Utility.Observability.cadence import CadenceConfig

from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=StereoFrame)


@T.runtime_checkable
class _FrontendWithResetStream(T.Protocol):
    def reset_stream(self) -> None: ...


@T.runtime_checkable
class _FrontendWithIMUWindow(T.Protocol):
    def set_imu_window(self, imu_window: IMUData | None, bias_ref: torch.Tensor | None = None) -> None: ...


@T.runtime_checkable
class _VisualMapWithGravity(T.Protocol):
    gravity: torch.Tensor


class MACVO(IOdometry[T_SensorFrame], ConfigTestable):
    # Type alias of callback hooks for MAC-VO system. Will be called by the system on
    # certain event occurs (optimization finish, for instance.)
    T_SYSHOOK = Callable[["MACVO",], None]
    
    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile, mapping,
        frontend        : Module.IFrontend, 
        motion_model    : Module.IMotionModel[T_SensorFrame],
        kp_selector     : Module.IKeypointSelector,
        map_selector    : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.ICovariance2to3,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector[T_SensorFrame],
        optimizer       : Module.IOptimizer,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")
        
        self.graph = VisualMap()
        self.config = SimpleNamespace(**_excessive_args)
        self.device = device
        self.mapping: bool = mapping
        self.match_cov_default: float = match_cov_default

        # Modules
        self.Frontend = frontend
        self.MotionEstimator = motion_model
        self.KeypointSelector = kp_selector
        self.MappointSelector = map_selector
        self.OutlierFilter = obs_filter
        self.ObsCovModel = obs_covmodel
        self.MapRefiner = post_process
        self.KeyframeSelector = kf_selector
        self.Optimizer = optimizer
        # end

        self.min_num_point = 10
        self.num_point = num_point
        self.edge_width = edgewidth
        self.isinitiated = False
        self.init_cfg: SimpleNamespace = getattr(self.config, "init", SimpleNamespace(method="identity", drt_window_len=10))
        self.init_method: str = getattr(self.init_cfg, "method", "identity")
        self._init_frames: list[T_SensorFrame] = []
        self._init_depths: list[Module.IStereoDepth.Output] = []
        self._init_fail_count: int = 0
        self._init_complete: bool = self.init_method != "drt_loose"
        self.g_W = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        T.cast(_VisualMapWithGravity, self.graph).gravity = self.g_W
        self._drt_initializer = None
        self._imu_encoder = None
        self._imu_since_keyframe: IMUData | None = None
        self.debug_logger: DebugLogger | None = None
        self._log_cadence = CadenceConfig()
        log_cfg = getattr(self.config, "logging", None)
        if log_cfg is not None and bool(getattr(log_cfg, "enabled", False)):
            self.debug_logger = DebugLogger.from_config(log_cfg, seed=int(getattr(log_cfg, "seed", 0)))
            self.debug_logger.assert_all_sinks_writable()
            local_cfg = getattr(log_cfg, "local", SimpleNamespace())
            self._log_cadence = CadenceConfig(
                dump_every=int(getattr(local_cfg, "dump_every", 200)),
                forced_first_n=int(getattr(local_cfg, "forced_first_n", 10)),
                on_anomaly=bool(getattr(local_cfg, "on_anomaly", True)),
            )
            setattr(self.Optimizer, "debug_logger", self.debug_logger)
        
        # Context for tracking
        # [0] - Frame Source Data
        # [1] - Frame index (in visual map)
        # [2] - Frame stereo depth
        self.prev_keyframe: tuple[T_SensorFrame, int, Module.IStereoDepth.Output] | None = None
        
        # Hooks
        self.on_optimize_writeback: list[MACVO.T_SYSHOOK] = []

        self.report_config()
    
    @classmethod
    def from_config(cls, cfg: SimpleNamespace):
        odomcfg = cfg.Odometry
        # Initialize modules for VO
        Frontend            = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        MotionEstimator     = Module.IMotionModel[T_SensorFrame].instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        MappointSelector    = Module.IKeypointSelector.instantiate(odomcfg.mappoint.type, odomcfg.mappoint.args)
        ObservationFilter   = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObserveCovModel     = Module.ICovariance2to3.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner          = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector    = Module.IKeyframeSelector[T_SensorFrame].instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer           = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)
        
        return cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            map_selector=MappointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            **vars(odomcfg.args),
        )
    
    def report_config(self):
        # Cute fine-print boxes
        box1 = Panel.fit(
            "\n".join(
                [
                    f"DepthEstimator cov: {self.Frontend.provide_cov[0]}",
                    f"MatchEstimator cov: {self.Frontend.provide_cov[1]}",
                    f"Observation cov:    {self.ObsCovModel.__class__.__name__}",
                ]
            ),
            title="Odometry Covariance",
            title_align="left",
        )
        box2 = Panel.fit(
            "\n".join(
                [   
                    f"Optimizer       -'{self.Optimizer       .__class__.__name__}'",
                    f"Frontend        -'{self.Frontend        .__class__.__name__}'",
                    f"MotionEstimator -'{self.MotionEstimator .__class__.__name__}'",
                    f"KeypointSelector-'{self.KeypointSelector.__class__.__name__}'",
                    f"MappointSelector-'{self.MappointSelector.__class__.__name__}'",
                    f"OutlierFilter   -'{self.OutlierFilter   .__class__.__name__}'",
                    f"MapRefiner      -'{self.MapRefiner      .__class__.__name__}'",
                ]
            ),
            title="Odometry Modules",
            title_align="left",
        )
        GlobalConsole.print(Columns([box1, box2]))

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        Module.IKeyframeSelector.is_valid_config(config.keyframe)
        Module.IMapProcessor.is_valid_config(config.postprocess)
        Module.IObservationFilter.is_valid_config(config.outlier)
        Module.IMotionModel.is_valid_config(config.motion)
        Module.IKeypointSelector.is_valid_config(config.keypoint)
        Module.ICovariance2to3.is_valid_config(config.cov.obs)
        Module.IFrontend.is_valid_config(config.frontend)
        Module.IOptimizer.is_valid_config(config.optimizer)
        
        cls._enforce_config_spec(config.args, {
            "device"            : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "num_point"         : lambda b: isinstance(b, int) and b > 0, 
            "edgewidth"         : lambda b: isinstance(b, int) and b > 0, 
            "match_cov_default" : lambda b: isinstance(b, (float, int)) and b > 0.0, 
            "profile"           : lambda b: isinstance(b, bool),
            "mapping"           : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)

    def initialize(self, frame0: T_SensorFrame):
        if self.init_method == "drt_loose":
            self._bootstrap_collect(frame0)
            return

        depth0          = self.Frontend.estimate_depth(frame0.stereo)
        est_pose        = self.MotionEstimator.predict(frame0, None, depth0.depth).unsqueeze(0)
        
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame0.stereo.T_BS,
            "vel"         : torch.zeros((1, 3), dtype=torch.float32),
            "bias_g"      : torch.zeros((1, 3), dtype=torch.float32),
            "bias_a"      : torch.zeros((1, 3), dtype=torch.float32),
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame0.stereo.frame_ns], dtype=torch.long),
            "K"           : frame0.stereo.K,
            "baseline"    : frame0.stereo.baseline,
        }))
        self.OutlierFilter.set_meta(frame0.stereo)
        self.prev_keyframe = (frame0, int(frame_idx.item()), depth0)
        self._imu_since_keyframe = None
        self.isinitiated = True

    def _bootstrap_collect(self, frame: T_SensorFrame) -> None:
        if not hasattr(frame, "imu"):
            Logger.write("warn", "DRT-loose requested but sequence has no IMU. Falling back to identity init.")
            self.init_method = "identity"
            self.initialize(frame)
            self.isinitiated = True
            return

        depth = self.Frontend.estimate_depth(frame.stereo)
        self._init_frames.append(frame)
        self._init_depths.append(depth)
        self.OutlierFilter.set_meta(frame.stereo)

        n_init = int(getattr(self.init_cfg, "drt_window_len", 10))
        if len(self._init_frames) < n_init:
            return

        if self._drt_initializer is None:
            from Module.Network.AirIMU.encoder import IMUEncoder
            from Module.Initialization.DRTLoose import DRTLooseInitializer

            imu_cfg = getattr(self.init_cfg, "imu", SimpleNamespace(
                ckpt_path=None,
                jacobian_eps=1e-5,
                sigma_repr_mode="diag",
                feature_dim=64,
                emit_jacobians=True,
            ))
            self._imu_encoder = IMUEncoder(imu_cfg)
            self._drt_initializer = DRTLooseInitializer(
                frontend=self.Frontend,
                imu_encoder=self._imu_encoder,
                cfg=self.init_cfg,
            )

        init_step = int(getattr(frame, "frame_idx", len(self._init_frames)))
        init_frames = T.cast(list[StereoInertialFrame], self._init_frames)
        init_out = self._drt_initializer.run(init_frames, self._init_depths, logger=self.debug_logger, log_step=init_step)
        if init_out.ok:
            self._write_init_to_map(init_out)
            self._init_complete = True
            self.isinitiated = True
            if isinstance(self.Frontend, _FrontendWithResetStream):
                self.Frontend.reset_stream()
            return

        self._init_fail_count += 1
        Logger.write("warn", f"DRT-loose init failed ({self._init_fail_count}): {init_out.reason}")
        if self.debug_logger is not None:
            self.debug_logger.log_scalars(
                {
                    "init.drt.window_len": float(len(self._init_frames)),
                    "init.drt.accept": 0.0,
                    "init.drt.scale_s": 1.0,
                    "init.drt.gW.mag": float(torch.linalg.vector_norm(init_out.gravity).item()) if init_out.gravity.numel() == 3 else 0.0,
                    "init.drt.gW.mag_err": float(abs(torch.linalg.vector_norm(init_out.gravity).item() - 9.81007)) if init_out.gravity.numel() == 3 else 9.81007,
                    "init.drt.bg.norm": float(torch.linalg.vector_norm(init_out.bias_g).item()) if init_out.bias_g.numel() == 3 else 0.0,
                    "init.drt.bg_solver.residual": float(init_out.diagnostics.get("bg_solver_residual", 0.0)) if init_out.diagnostics else 0.0,
                    "init.drt.n_views.total": float(init_out.diagnostics.get("n_views_total", 0.0)) if init_out.diagnostics else 0.0,
                    "init.drt.parallax_px.p50": float(init_out.diagnostics.get("parallax_p50", 0.0)) if init_out.diagnostics else 0.0,
                    "init.drt.parallax_px.p5": float(init_out.diagnostics.get("parallax_p5", 0.0)) if init_out.diagnostics else 0.0,
                },
                init_step,
            )
        self._init_frames.pop(0)
        self._init_depths.pop(0)

        if self._init_fail_count > 2 * n_init:
            Logger.write("warn", "DRT-loose fallback to identity init.")
            self.init_method = "identity"
            depth0 = self.Frontend.estimate_depth(frame.stereo)
            est_pose = self.MotionEstimator.predict(frame, None, depth0.depth).unsqueeze(0)
            frame_idx = self.graph.frames.push(FrameNode.init({
                "pose"        : est_pose,
                "T_BS"        : frame.stereo.T_BS,
                "vel"         : torch.zeros((1, 3), dtype=torch.float32),
                "bias_g"      : torch.zeros((1, 3), dtype=torch.float32),
                "bias_a"      : torch.zeros((1, 3), dtype=torch.float32),
                "need_interp" : torch.tensor([0], dtype=torch.bool),
                "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
                "K"           : frame.stereo.K,
                "baseline"    : frame.stereo.baseline,
            }))
            self.prev_keyframe = (frame, int(frame_idx.item()), depth0)
            self._imu_since_keyframe = None
            self._init_complete = True
            self.isinitiated = True

    def _push_imu_edge(self, from_idx: int, to_idx: int, imu_pre) -> None:
        self.graph.imu_edges.push(IMUEdgeNode.init({
            "from_frame": torch.tensor([from_idx], dtype=torch.long),
            "to_frame"  : torch.tensor([to_idx], dtype=torch.long),
            "delta_R"   : imu_pre.delta_R.unsqueeze(0).to(torch.float64),
            "delta_v"   : imu_pre.delta_v.unsqueeze(0).to(torch.float64),
            "delta_p"   : imu_pre.delta_p.unsqueeze(0).to(torch.float64),
            "Sigma"     : imu_pre.Sigma.unsqueeze(0).to(torch.float64),
            "dt"        : imu_pre.dt.view(1).to(torch.float64),
            "bias_ref"  : imu_pre.bias_ref.unsqueeze(0).to(torch.float64),
            "J_R_bg"    : imu_pre.J_R_bg.unsqueeze(0).to(torch.float64),
            "J_v_bg"    : imu_pre.J_v_bg.unsqueeze(0).to(torch.float64),
            "J_v_ba"    : imu_pre.J_v_ba.unsqueeze(0).to(torch.float64),
            "J_p_bg"    : imu_pre.J_p_bg.unsqueeze(0).to(torch.float64),
            "J_p_ba"    : imu_pre.J_p_ba.unsqueeze(0).to(torch.float64),
        }))

    def _write_init_to_map(self, init_out) -> None:
        self.graph = VisualMap()
        T.cast(_VisualMapWithGravity, self.graph).gravity = init_out.gravity.to(torch.float64)
        for k, frame in enumerate(self._init_frames):
            R_B = init_out.rotation[k].double()
            p_B = init_out.position[k].double()
            q_B = T.cast(pp.LieTensor, pp.mat2SO3(R_B.unsqueeze(0))).tensor()[0]
            T_WB = pp.SE3(torch.cat([p_B, q_B], dim=-1).unsqueeze(0))
            T_BS = pp.SE3(frame.stereo.T_BS).double()
            T_WC = T.cast(pp.LieTensor, (T_WB @ T_BS).float())
            self.graph.frames.push(FrameNode.init({
                "pose"        : T_WC.tensor(),
                "T_BS"        : frame.stereo.T_BS,
                "vel"         : init_out.velocity[k].view(1, 3).float(),
                "bias_g"      : init_out.bias_g.view(1, 3).float(),
                "bias_a"      : init_out.bias_a.view(1, 3).float(),
                "need_interp" : torch.tensor([0], dtype=torch.bool),
                "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
                "K"           : frame.stereo.K,
                "baseline"    : frame.stereo.baseline,
            }))

        for k in range(len(init_out.imu_pres)):
            self._push_imu_edge(k, k + 1, init_out.imu_pres[k])

        self.prev_keyframe = (
            self._init_frames[-1],
            len(self._init_frames) - 1,
            self._init_depths[-1],
        )
        self._imu_since_keyframe = None
        self.g_W = init_out.gravity.to(torch.float64)
        T.cast(_VisualMapWithGravity, self.graph).gravity = self.g_W
        Logger.write(
            "info",
            f"DRT-loose init OK: N_init={len(self._init_frames)}, "
            f"|b_g|={init_out.bias_g.norm():.4g}, |g|={init_out.gravity.norm():.4g}",
        )
        if self.debug_logger is not None:
            step = int(getattr(self._init_frames[-1], "frame_idx", len(self._init_frames)))
            drt_dump = {
                "accept": torch.tensor([1], dtype=torch.bool),
                "n_views": torch.tensor([init_out.diagnostics.get("n_views_total", 0) if init_out.diagnostics else 0], dtype=torch.int64),
                "parallax_px_med": torch.tensor([init_out.diagnostics.get("parallax_p50", 0.0) if init_out.diagnostics else 0.0], dtype=torch.float32),
                "bg_final": init_out.bias_g.float(),
                "gW": init_out.gravity.float(),
                "scale_s": torch.tensor([1.0], dtype=torch.float32),
            }
            self.debug_logger.dump_artifact("dumps", collect_runtime_dump(
                step=step,
                epoch=0,
                seq_id="bootstrap",
                frame_idx=step,
                batch_idx=0,
                cadence="forced_early",
                drt=drt_dump,
            ), step)

    @staticmethod
    def _append_imu_window(base: IMUData | None, window: IMUData | None) -> IMUData | None:
        if window is None:
            return base
        if base is None:
            return window

        base_time = base.time_ns[..., 0] if base.time_ns.ndim == 3 else base.time_ns
        window_time = window.time_ns[..., 0] if window.time_ns.ndim == 3 else window.time_ns

        append_start = 0
        if base_time.shape[0] == 1 and window_time.shape[0] == 1:
            append_start = int(torch.searchsorted(window_time[0], base_time[0, -1], right=True).item())
        elif torch.equal(base_time[:, -1:], window_time[:, :1]):
            append_start = 1

        if append_start >= window_time.shape[1]:
            return base

        return IMUData(
            T_BS=base.T_BS,
            time_ns=torch.cat([base.time_ns, window.time_ns[:, append_start:]], dim=1),
            gravity=base.gravity,
            acc=torch.cat([base.acc, window.acc[:, append_start:, :]], dim=1),
            gyro=torch.cat([base.gyro, window.gyro[:, append_start:, :]], dim=1),
        )

    @staticmethod
    def _imu_sample_count(imu: IMUData | None) -> int:
        if imu is None:
            return 0
        if imu.time_ns.ndim < 2 or imu.time_ns.shape[0] == 0:
            return 0
        return int(imu.time_ns.shape[1])

    def _insert_runtime_imu_edge(
        self,
        frame1: T_SensorFrame,
        prev_frame_idx: torch.Tensor,
        frame_idx: torch.Tensor,
        step: int,
    ) -> None:
        imu = self._imu_since_keyframe
        if imu is not None and self._imu_sample_count(imu) >= 2:
            if self._imu_encoder is None:
                from Module.Network.AirIMU.encoder import IMUEncoder
                imu_cfg = getattr(self.init_cfg, "imu", SimpleNamespace(
                    ckpt_path=None,
                    jacobian_eps=1e-5,
                    sigma_repr_mode="diag",
                    feature_dim=64,
                    emit_jacobians=True,
                ))
                self._imu_encoder = IMUEncoder(imu_cfg)
            bias_ref = torch.cat(
                [
                    self.graph.frames.data["bias_g"][prev_frame_idx].float(),
                    self.graph.frames.data["bias_a"][prev_frame_idx].float(),
                ],
                dim=-1,
            )
            corr = narrow_corrector_output(self._imu_encoder.corrector.inference({"acc": imu.acc, "gyro": imu.gyro}))
            pre = self._imu_encoder.preint(
                corrected_acc=imu.acc + corr["correction_acc"],
                corrected_gyro=imu.gyro + corr["correction_gyro"],
                acc_cov=corr["cov_state"]["acc_cov"],
                gyro_cov=corr["cov_state"]["gyro_cov"],
                dt=imu.time_delta.float() * 1e-9,
                bias_ref=bias_ref,
                emit_jacobians=True,
                logger=self.debug_logger,
                log_step=step,
            )
            self._push_imu_edge(int(prev_frame_idx.item()), int(frame_idx.item()), SimpleNamespace(
                delta_R=pre.delta_R[0].cpu(),
                delta_v=pre.delta_v[0].cpu(),
                delta_p=pre.delta_p[0].cpu(),
                Sigma=pre.Sigma[0].cpu(),
                dt=pre.dt_total[0].cpu(),
                bias_ref=pre.bias_ref[0].cpu(),
                J_R_bg=pre.J_R_bg[0].cpu() if pre.J_R_bg is not None else torch.zeros((3, 3)),
                J_v_bg=pre.J_v_bg[0].cpu() if pre.J_v_bg is not None else torch.zeros((3, 3)),
                J_v_ba=pre.J_v_ba[0].cpu() if pre.J_v_ba is not None else torch.zeros((3, 3)),
                J_p_bg=pre.J_p_bg[0].cpu() if pre.J_p_bg is not None else torch.zeros((3, 3)),
                J_p_ba=pre.J_p_ba[0].cpu() if pre.J_p_ba is not None else torch.zeros((3, 3)),
            ))
        elif hasattr(frame1, "imu"):
            imu_n = self._imu_sample_count(imu)
            from_idx = int(prev_frame_idx.item())
            to_idx = int(frame_idx.item())
            if imu_n == 0:
                Logger.write("warn", f"Missing IMU window for keyframe pair {from_idx}->{to_idx}, skipping IMU edge")
            else:
                Logger.write("warn", f"Insufficient IMU samples ({imu_n}) for keyframe pair {from_idx}->{to_idx}, skipping IMU edge")

    def run_pair(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        assert self.prev_keyframe is not None
        step = int(getattr(frame1, "frame_idx", len(self.graph.frames)))
        self._imu_since_keyframe = self._append_imu_window(
            self._imu_since_keyframe, getattr(frame1, "imu", None)
        )
        
        # Check if current frame is the keyframe ########################################
        if not self.KeyframeSelector.isKeyframe(frame1):            
            self.push_keyframe(frame1, self.graph.frames.data["pose"][self.prev_keyframe[1]].unsqueeze(0), need_interp=True)
            return
        
        depth0 = self.prev_keyframe[2]
        if isinstance(self.Frontend, _FrontendWithIMUWindow):
            bias_ref = torch.cat(
                [
                    self.graph.frames.data["bias_g"][self.prev_keyframe[1]].float(),
                    self.graph.frames.data["bias_a"][self.prev_keyframe[1]].float(),
                ],
                dim=-1,
            ).unsqueeze(0) if len(self.graph.frames) > 0 else None
            self.Frontend.set_imu_window(self._imu_since_keyframe, bias_ref=bias_ref)
        if self.debug_logger is None:
            depth1, match01 = self.Frontend.estimate_pair(frame0.stereo, frame1.stereo)
        else:
            with self.debug_logger.phase("frontend", step):
                depth1, match01 = self.Frontend.estimate_pair(frame0.stereo, frame1.stereo)

        # Receive optimization result from previous step (if exists) ####################
        # NOTE: should always writeback optimized pose to global map before selecting new 
        # keypoints (register new 3D point) on that frame.
        self.Optimizer.write_map(self.graph)
        for func in self.on_optimize_writeback: func(self)
        
        # Motion model provide an initial guess to the pose of frame1 ###################
        # Update motion model (this must be after write_back to get latest result)
        # NOTE: I assume the motion estimator works on stereo camera frame (not body frame)
        self.MotionEstimator.update(pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]]))
        est_pose = self.MotionEstimator.predict(frame1, match01.flow, depth1.depth).unsqueeze(0)
        
        # Generate Keypoints for frame 0 and 1 ##########################################
        kp0_uv  = self.KeypointSelector.select_point(frame0.stereo, self.num_point, depth0, depth1, match01)
        kp1_uv  = kp0_uv + self.Frontend.retrieve_pixels(kp0_uv, match01.flow).T
        
        inbound_mask= filterPointsInRange(
            kp1_uv, 
            (self.edge_width, frame1.stereo.width - self.edge_width), 
            (self.edge_width, frame1.stereo.height - self.edge_width)
        )
        kp0_uv  = kp0_uv[inbound_mask]
        kp1_uv  = kp1_uv[inbound_mask]
        
        # Retrieve depth and depth cov for kp on frame 0 and 1 ##########################
        kp0_d               = self.Frontend.retrieve_pixels(kp0_uv, depth0.depth).squeeze(0)
        kp0_disparity       = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity)
        kp0_sigma_disparity = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity_uncertainty)
        kp0_sigma_dd        = self.Frontend.retrieve_pixels(kp0_uv, depth0.cov)
        kp0_sigma_dd        = kp0_sigma_dd.squeeze(0) if kp0_sigma_dd is not None else None
        
        kp1_d               = self.Frontend.retrieve_pixels(kp1_uv, depth1.depth).squeeze(0)
        kp1_disparity       = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity)
        kp1_sigma_disparity = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity_uncertainty)
        kp1_sigma_dd        = self.Frontend.retrieve_pixels(kp1_uv, depth1.cov)
        kp1_sigma_dd        = kp1_sigma_dd.squeeze(0) if kp1_sigma_dd is not None else None
        
        
        # Retrieve match cov for kp on frame 0 and 1    #################################
        num_kp = kp0_uv.size(0)
        
        # kp 0 has a fake sigma uv as it is manually selected pixels. This UV 
        # represents the uncertainty introduced by the quantization process when 
        # taking photo with discrete pixels.
        kp0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
        kp0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.
        
        kp1_sigma_uv = self.Frontend.retrieve_pixels(kp0_uv, match01.cov)
        kp1_sigma_uv = kp1_sigma_uv.T if kp1_sigma_uv is not None else None
        
        # Record color of keypoints (for visualization) #################################
        kp0_uv_cpu = kp0_uv.cpu()
        kp0_color  = frame0.stereo.imageL[..., kp0_uv_cpu[..., 1], kp0_uv_cpu[..., 0]].squeeze(0).T
        kp0_color  = (kp0_color * 255).to(torch.uint8)
        
        # Project from 2D -> 3D #########################################################
        pos0_Tc = pixel2point_NED(kp0_uv, kp0_d, frame0.stereo.frame_K).cpu()
        pos0_covTc  = self.ObsCovModel.estimate(frame0.stereo, kp0_uv, depth0, kp0_sigma_dd, kp0_sigma_uv)
        pos1_covTc  = self.ObsCovModel.estimate(frame1.stereo, kp1_uv, depth1, kp1_sigma_dd, kp1_sigma_uv)
        
        
        # Run Outlier Filter ############################################################
        match_obs = MatchObs.init({
            "pixel1_uv"      : kp0_uv_cpu,
            "pixel2_uv"      : kp1_uv.cpu(),
            
            "pixel1_d"       : kp0_d.unsqueeze(-1).cpu(),
            "pixel2_d"       : kp1_d.unsqueeze(-1).cpu(),
            
            "pixel1_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp0_disparity is None else kp0_disparity.T.cpu(),
            "pixel2_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp1_disparity is None else kp1_disparity.T.cpu(),
            
            "pixel1_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_disparity is None else kp0_sigma_disparity.T.cpu(),
            "pixel2_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_disparity is None else kp1_sigma_disparity.T.cpu(),
            
            "pixel1_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_dd is None else kp0_sigma_dd.unsqueeze(-1).cpu(),
            "pixel2_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_dd is None else kp1_sigma_dd.unsqueeze(-1).cpu(),
            
            "pixel1_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp0_sigma_uv is None else kp0_sigma_uv,
            "pixel2_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp1_sigma_uv is None else kp1_sigma_uv,
            
            "obs1_covTc"     : pos0_covTc,
            "obs2_covTc"     : pos1_covTc,
            "c"              : (
                torch.ones((num_kp, 1), dtype=torch.float32)
                if not hasattr(match01, "c") or getattr(match01, "c") is None
                else self.Frontend.retrieve_pixels(kp0_uv, getattr(match01, "c")).T.float().cpu()
            ),
        })
        assert self.OutlierFilter.verify_shape(match_obs), "The provided MatchFactor does not contain all data for outlier filter."
        mask = self.OutlierFilter.filter(match_obs, torch.device("cpu"))
        match_obs = match_obs[mask]
        
        # Register the factor graph #####################################################
        prev_pose       = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
        prev_rot        = prev_pose.rotation().matrix().repeat((num_kp, 1, 1)).to(torch.float64)
        num_match_orig  = len(self.graph.match)
        
        point_idx = self.graph.points.push(PointNode.init({
            "pos_Tw": pp.SE3_type.Act(prev_pose, pos0_Tc)[..., :3],  # NOTE: Refer to https://github.com/pypose/pypose/issues/342 
            "cov_Tw": torch.bmm(torch.bmm(prev_rot, pos0_covTc), prev_rot.transpose(1, 2)),
            "color" : kp0_color
        })[mask])
        frame_idx      = self.push_keyframe(frame1, est_pose)
        prev_frame_idx = torch.tensor([self.prev_keyframe[1]], dtype=torch.long)
        match_idx      = self.graph.match.push(match_obs)
        
        num_match_kp = len(match_obs)
        self.graph.point2match.add(point_idx, match_idx)    # Associate point -> match
        self.graph.match2point.set(match_idx, point_idx)    # Associate match -> point
        self.graph.frame2match.add(prev_frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.frame2match.add(frame_idx     , torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.match2frame1.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(prev_frame_idx.item()))    # Associate match -> frame1
        self.graph.match2frame2.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(frame_idx.item()     ))    # Associate match -> frame2
        self._insert_runtime_imu_edge(frame1, prev_frame_idx, frame_idx, step)

        # Visualization #################################################################
        fig_plt.plot_imatcher("matching", match01, frame0, frame1)
        fig_plt.plot_istereo ("stereo_d", depth1 , frame1)
        fig_plt.plot_macvo   ("macvo_kp", match_obs, depth1, match01, frame0, frame1)

        # Update the tracking context ###################################################
        self.prev_keyframe = (frame1, int(frame_idx.item()), depth1)
        self._imu_since_keyframe = None

        # Launch Optimization task  #####################################################
        if match_idx.size(0) < self.min_num_point:
            # NOTE: if lost track, we do not do mapping since the pose is not reliable anyway.
            Logger.write("warn", f"VOLostTrack @ {frame1.frame_idx} - only get {match_idx.size(0)} observations")
            self.graph.frames.data["need_interp"][frame_idx] = True
            return
        else:
            if self.debug_logger is None:
                self.Optimizer.start_optimize(
                    self.Optimizer.get_graph_data(self.graph, frame_idx)
                )
            else:
                with self.debug_logger.phase("backend", step):
                    self.Optimizer.start_optimize(
                        self.Optimizer.get_graph_data(self.graph, frame_idx)
                    )
        
        # Add (dense) mapping points to the map #########################################
        if self.mapping:
            map0_uv       = self.MappointSelector.select_point(frame0.stereo, 2000, depth0, depth1, match01)
            num_kp        = map0_uv.size(0)
            map0_d        = self.Frontend.retrieve_pixels(map0_uv, depth0.depth).squeeze(0)
            map0_Tc       = pixel2point_NED(map0_uv, map0_d, frame0.stereo.frame_K).cpu()
            
            map0_sigma_dd = self.Frontend.retrieve_pixels(map0_uv, depth0.cov)
            map0_sigma_dd = map0_sigma_dd.squeeze(0) if (map0_sigma_dd is not None) else None
            map0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
            map0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.
            map0_Tc_cov = self.ObsCovModel.estimate(frame0.stereo, map0_uv, depth0, map0_sigma_dd, map0_sigma_uv)
            
            map0_uv_cpu = map0_uv.cpu()
            map0_color  = frame0.stereo.imageL[..., map0_uv_cpu[..., 1], map0_uv_cpu[..., 0]].squeeze(0).T
            map0_color  = (map0_color * 255).to(torch.uint8)
            
            num_map_orig  = len(self.graph.map_points)
            num_mappoint  = map0_Tc.size(0)
            map_idx = self.graph.map_points.push(PointNode.init({
                "pos_Tw": pp.SE3_type.Act(prev_pose, map0_Tc)[..., :3],
                "cov_Tw": map0_Tc_cov,
                "color" : map0_color,
            }))
            self.graph.frame2map.add(frame_idx, torch.tensor([num_map_orig], dtype=torch.long), torch.tensor([num_mappoint], dtype=torch.long))   # Associate frame -> map

        if self.debug_logger is not None and should_dump(step, self._log_cadence):
            imu_window = getattr(frame1, "imu", None)
            data_dump = {
                "image_L": frame1.stereo.imageL[0].detach().cpu(),
                "image_R": frame1.stereo.imageR[0].detach().cpu(),
                "K": frame1.stereo.K[0].detach().cpu(),
                "T_BS": frame1.stereo.T_BS[0].detach().cpu(),
            }
            if imu_window is not None:
                data_dump["imu_window"] = torch.cat([imu_window.acc[0], imu_window.gyro[0]], dim=-1).detach().cpu()
                data_dump["imu_dts"] = (imu_window.time_delta[0].float() * 1e-9).detach().cpu()
            frontend_dump = {
                "depth": depth1.depth[0].detach().cpu(),
                "disparity": depth1.disparity[0].detach().cpu() if depth1.disparity is not None else torch.empty((1,)),
                "flow": match01.flow[0].detach().cpu(),
                "flow_cov": match01.cov[0].detach().cpu() if match01.cov is not None else torch.empty((1,)),
            }
            dump_payload = collect_runtime_dump(
                step=step,
                epoch=0,
                seq_id=str(getattr(frame1, "seq_id", "unknown")),
                frame_idx=step,
                batch_idx=0,
                cadence=cadence_reason(step, self._log_cadence),
                data=data_dump,
                frontend=frontend_dump,
            )
            self.debug_logger.dump_artifact("dumps", dump_payload, step)
        if self.debug_logger is not None:
            self.debug_logger.on_step_end(step)

    def push_keyframe(self, frame: T_SensorFrame, est_pose: pp.LieTensor | torch.Tensor, need_interp: bool=False) -> torch.Tensor:
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.stereo.T_BS,
            "vel"         : torch.zeros((1, 3), dtype=torch.float32),
            "bias_g"      : torch.zeros((1, 3), dtype=torch.float32),
            "bias_a"      : torch.zeros((1, 3), dtype=torch.float32),
            "need_interp" : torch.tensor([need_interp], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
            "K"           : frame.stereo.K,
            "baseline"    : frame.stereo.baseline,
        }))
        return frame_idx

    @Timer.cpu_timeit("Odom_Runtime")
    @Timer.gpu_timeit("Odom_Runtime")
    def run(self, frame: T_SensorFrame) -> None:
        """
        The main process that continuously running to manage different modules in MAC-VO.
        The multi-threading part will be managed in this function.
        Args:
            frame (T_SensorFrame): The current stereo frame to be processed.
        Returns:
            None
        """

        if not self.isinitiated:
            self.initialize(frame)
            return
        
        assert self.prev_keyframe is not None
        self.run_pair(self.prev_keyframe[0], frame)

    def get_map(self) -> VisualMap:
        return self.graph

    def terminate(self) -> None:
        super().terminate()
        if self.prev_keyframe is not None:
            self.Optimizer.write_map(self.graph)
        self.Optimizer.terminate()
        self.MapRefiner.elaborate_map(self.graph.frames)
        if self.debug_logger is not None:
            self.debug_logger.close()

    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)

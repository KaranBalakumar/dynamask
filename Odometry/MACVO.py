import torch
import pypose as pp
import typing as T
from types import SimpleNamespace

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import StereoFrame
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Timer import Timer
from Utility.Visualize import fig_plt
from Utility.Extensions import ConfigTestable

from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=StereoFrame)
_EDN2NED_ROT = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=torch.float64)


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
        init_cfg        = None,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")

        self.graph = VisualMap()
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
        self._init_cfg = init_cfg
        self._drt_init_buffer: list = []
        self.imu_context = None   # set by from_config if airio block present
        
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
        
        # Build IMUContext (AirIO + EKF) if configured
        imu_ctx = None
        cfg_init = getattr(odomcfg, "init", None)
        if cfg_init is not None and getattr(cfg_init, "enabled", False):
            airio_cfg_node = getattr(cfg_init, "airio", None)
            if airio_cfg_node is not None:
                from types import SimpleNamespace as _SNS
                from Module.Network.IMUContext import IMUContext
                model_cfg = _SNS(propcov=getattr(airio_cfg_node, "propcov", True))
                imu_ctx = IMUContext(
                    airio_cfg=model_cfg,
                    airio_ckpt=getattr(airio_cfg_node, "ckpt", None),
                    gravity=getattr(cfg_init, "gravity_norm", 9.81007),
                )
                Logger.write("info", "IMUContext (AirIO + EKF) initialised")

        system = cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            map_selector=MappointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            init_cfg=getattr(odomcfg, "init", None),
            **vars(odomcfg.args),
        )
        system.imu_context = imu_ctx
        return system

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
        })

    def initialize(self, frame0: T_SensorFrame):
        depth0          = self.Frontend.estimate_depth(frame0.stereo)
        est_pose        = self.MotionEstimator.predict(frame0, None, depth0.depth).unsqueeze(0)
        
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame0.stereo.T_BS,
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame0.stereo.frame_ns], dtype=torch.long),
            "K"           : frame0.stereo.K,
            "baseline"    : frame0.stereo.baseline,
        }))
        self.OutlierFilter.set_meta(frame0.stereo)
        self.prev_keyframe = (frame0, int(frame_idx.item()), depth0)

    def run_pair(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        assert self.prev_keyframe is not None
        
        # Check if current frame is the keyframe ########################################
        if not self.KeyframeSelector.isKeyframe(frame1):            
            self.push_keyframe(frame1, self.graph.frames.data["pose"][self.prev_keyframe[1]].unsqueeze(0), need_interp=True)
            return
        
        depth0          = self.prev_keyframe[2]
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

        # Visualization #################################################################
        fig_plt.plot_imatcher("matching", match01, frame0, frame1)
        fig_plt.plot_istereo ("stereo_d", depth1 , frame1)
        fig_plt.plot_macvo   ("macvo_kp", match_obs, depth1, match01, frame0, frame1)

        # Update the tracking context ###################################################
        self.prev_keyframe = (frame1, int(frame_idx.item()), depth1)

        # Launch Optimization task  #####################################################
        if match_idx.size(0) < self.min_num_point:
            # NOTE: if lost track, we do not do mapping since the pose is not reliable anyway.
            Logger.write("warn", f"VOLostTrack @ {frame1.frame_idx} - only get {match_idx.size(0)} observations")
            self.graph.frames.data["need_interp"][frame_idx] = True
            return
        else:
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

    def _handle_init(self, frame: T_SensorFrame) -> None:
        """Route to DRT init or heuristic init based on config."""
        cfg_init = self._init_cfg
        use_drt = (cfg_init is not None
                   and getattr(cfg_init, "enabled", False)
                   and hasattr(frame, "imu"))

        if use_drt:
            self._drt_init_buffer.append(frame)
            n_needed = getattr(cfg_init, "min_keyframes", 10)
            # Require n_needed temporally-spaced keyframes (≥220ms apart, matching
            # the C++ addFeatureCheckParallax threshold). Count selected frames directly
            # rather than using a coverage heuristic, since at 20Hz each "step" is
            # 250ms (5 frames) rather than exactly 220ms.
            buf = self._drt_init_buffer
            if len(buf) < 2:
                return
            _MIN_GAP_NS = 220_000_000
            _n_sel, _last_ns = 1, buf[0].stereo.frame_ns
            for _fr in buf[1:]:
                if _fr.stereo.frame_ns - _last_ns >= _MIN_GAP_NS:
                    _n_sel += 1
                    _last_ns = _fr.stereo.frame_ns
            if _n_sel < n_needed:
                return  # still accumulating

            from Module.Initialization.DRTLoose.bootstrap import run_drt_with_retry
            from Module.Initialization.DRTLoose.types import DRTInitConfig
            drt_cfg = DRTInitConfig(
                min_keyframes=n_needed,
                max_attempts=getattr(getattr(cfg_init, "retry", None), "max_attempts", 2),
                window_scales=tuple(getattr(getattr(cfg_init, "retry", None), "window_scale", [1.0, 1.4, 1.8])),
                fallback=getattr(cfg_init, "fallback", "heuristic"),
                gravity_norm=getattr(cfg_init, "gravity_norm", 9.81007),
            )

            try:
                bootstrap = self._build_drt_bootstrap(drt_cfg)
                drt_result = run_drt_with_retry(drt_cfg, solver_fn=lambda scale: bootstrap.solve())
            except Exception as exc:
                Logger.write("warn", f"DRT bootstrap error ({exc}); using heuristic")
                drt_result = None

            if drt_result is not None and drt_result.success:
                Logger.write("info", "DRT init succeeded — seeding EKF from DRT result")
                self._seed_from_drt(drt_result, self._drt_init_buffer[0])
                self._drt_init_buffer.clear()
                self.isinitiated = True
            elif drt_result is not None and drt_result.retry_recommended:
                # Slide the window: drop oldest frame and wait for the next one.
                # This handles LOW_PARALLAX (stationary window) without giving up.
                reason = drt_result.failure_reason
                Logger.write("info", f"DRT init retry ({reason}); sliding window")
                self._drt_init_buffer.pop(0)
                # isinitiated stays False — next frame will trigger another attempt
            else:
                reason = drt_result.failure_reason if drt_result is not None else "exception"
                Logger.write("warn", f"DRT init failed ({reason}); using heuristic")
                self.initialize(self._drt_init_buffer[0])
                self._drt_init_buffer.clear()
                self.isinitiated = True
        else:
            self.initialize(frame)
            self.isinitiated = True

    def _build_drt_bootstrap(self, drt_cfg) -> "DRTLooseBootstrap":
        """Run FlowFormer on the init buffer and return a populated DRTLooseBootstrap.

        For each consecutive pair in the buffer:
          - estimate_depth + CovAwareSelector to select/track keypoints via optical flow
          - collect IMU ticks into bootstrap._imu_segments

        Visual rotations for the gyro-bias step are left as identity (sufficient for
        initial gyro-bias estimation when angular rate is small); the LiGT translation
        step uses real pixel observations from the flow tracker.
        """
        from Module.Initialization.DRTLoose.bootstrap import DRTLooseBootstrap
        from Module.Initialization.DRTLoose.tracks import FeatureTrack

        buf = self._drt_init_buffer
        frame0 = buf[0]

        R_BC, t_BC, corrected = self._resolve_drt_extrinsic(frame0)
        if corrected:
            Logger.write(
                "info",
                "DRT bootstrap: empty IMU T_BS detected; applying EDN→NED extrinsic correction",
            )

        # Camera intrinsics for pixel→normalized coordinate conversion
        K = frame0.stereo.K.squeeze(0).to(torch.float64)   # (3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        bootstrap = DRTLooseBootstrap(
            drt_cfg, R_BC=R_BC, t_BC=t_BC,
            gravity_norm=drt_cfg.gravity_norm,
        )

        # ── frame 0: select seed keypoints ──────────────────────────────────
        depth0  = self.Frontend.estimate_depth(frame0.stereo)
        kp0_uv  = self.KeypointSelector.select_point(
            frame0.stereo, self.num_point, depth0, depth0, None
        )                                                    # (M, 2) float32

        def _px_to_norm(uv: torch.Tensor) -> torch.Tensor:
            """Convert pixel (u,v) to normalized image coords ((u-cx)/fx, (v-cy)/fy)."""
            uv = uv.cpu().to(torch.float64)
            return torch.stack([
                (uv[..., 0] - cx) / fx,
                (uv[..., 1] - cy) / fy,
            ], dim=-1)

        # Temporal subsampling: select init keyframes ≥220ms apart (matches C++
        # addFeatureCheckParallax threshold of 0.22s). Optical flow is tracked
        # through every consecutive pair for quality; observations and IMU are
        # only accumulated at/between selected keyframes.
        MIN_FRAME_GAP_NS = 220_000_000  # 0.22s in nanoseconds
        selected_indices = [0]
        last_selected_ns = buf[0].stereo.frame_ns
        for _i in range(1, len(buf)):
            if buf[_i].stereo.frame_ns - last_selected_ns >= MIN_FRAME_GAP_NS:
                selected_indices.append(_i)
                last_selected_ns = buf[_i].stereo.frame_ns
        selected_set = set(selected_indices)
        buf_to_kf: dict[int, int] = {bi: ki for ki, bi in enumerate(selected_indices)}

        # track_obs stores NORMALIZED image coords so _bearing() works correctly.
        # Keys are selected-keyframe indices (0..n_selected-1), not raw buffer indices.
        track_obs: dict[int, dict[int, T.Any]] = {
            j: {0: _px_to_norm(kp0_uv[j])}
            for j in range(kp0_uv.shape[0])
        }
        alive_ids  = list(range(kp0_uv.shape[0]))
        prev_kp_uv = kp0_uv                                 # (M, 2) float32

        bootstrap._keyframe_timestamps.append(frame0.stereo.frame_ms * 1e-3)

        prev_frame = frame0
        current_segment: list = []  # IMU ticks accumulated toward the next selected KF

        for i in range(1, len(buf)):
            frame_i = buf[i]

            # ── run FlowFormer on every consecutive pair (tracking continuity) ──
            _, match_i = self.Frontend.estimate_pair(prev_frame.stereo, frame_i.stereo)

            # ── track surviving keypoints via optical flow ───────────────────
            if alive_ids:
                flow_at_kp  = self.Frontend.retrieve_pixels(prev_kp_uv, match_i.flow)  # (2, M)
                new_kp_uv   = prev_kp_uv + flow_at_kp.T                                # (M, 2)

                H, W = frame_i.stereo.height, frame_i.stereo.width
                ew   = self.edge_width
                inbound = (
                    (new_kp_uv[:, 0] >= ew) & (new_kp_uv[:, 0] < W - ew) &
                    (new_kp_uv[:, 1] >= ew) & (new_kp_uv[:, 1] < H - ew)
                )

                new_alive  = [tid for k, tid in enumerate(alive_ids) if inbound[k]]
                new_kp_uv  = new_kp_uv[inbound]

                # Record observation only at selected keyframe indices
                if i in selected_set:
                    ki = buf_to_kf[i]
                    for k, tid in enumerate(new_alive):
                        track_obs[tid][ki] = _px_to_norm(new_kp_uv[k])

                alive_ids  = new_alive
                prev_kp_uv = new_kp_uv

            # ── accumulate IMU ticks for the segment ending at the next KF ───
            imu = getattr(frame_i, "imu", None)
            if imu is not None:
                dt_ns = imu.time_delta[0, :, 0].to(torch.float64)  # (N-1,) ns
                gyro  = imu.gyro[0].to(torch.float64)               # (N, 3)
                acc   = imu.acc[0].to(torch.float64)                # (N, 3)
                for t in range(dt_ns.shape[0]):
                    current_segment.append((gyro[t], acc[t], dt_ns[t].item() * 1e-9))

            # When we reach a selected keyframe, flush the accumulated segment
            if i in selected_set:
                bootstrap._keyframe_timestamps.append(frame_i.stereo.frame_ms * 1e-3)
                bootstrap._imu_segments.append(current_segment)
                current_segment = []

            prev_frame = frame_i

        # ── build FeatureTrack list (≥2 observations required for LiGT) ─────
        bootstrap._tracks = [
            FeatureTrack(track_id=tid, obs=obs)
            for tid, obs in track_obs.items()
            if len(obs) >= 2
        ]
        Logger.write("info",
            f"DRT bootstrap: {len(buf)} buf frames → {len(selected_indices)} KFs selected "
            f"({len(bootstrap._imu_segments)} IMU segments, "
            f"{len(bootstrap._tracks)} tracks, alive={len(alive_ids)})"
        )

        # ── Stereo depth maps for metric translation recovery ──────────────
        depth_maps: list[torch.Tensor] = []
        for si in selected_indices:
            d_out = self.Frontend.estimate_depth(buf[si].stereo)
            depth_maps.append(d_out.depth.cpu())  # (1, 1, H, W)
        bootstrap.set_stereo_depth(depth_maps, K)

        return bootstrap

    @staticmethod
    def _resolve_drt_extrinsic(frame: T_SensorFrame) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return body→camera extrinsic for DRT with EuRoC empty-IMU convention handling."""
        T_BS = frame.stereo.T_BS
        R_BC = T_BS.rotation().matrix().squeeze(0).to(torch.float64)
        t_BC = T_BS.translation().squeeze(0).to(torch.float64)

        imu = getattr(frame, "imu", None)
        imu_t_bs = getattr(imu, "T_BS", None)
        if imu_t_bs is not None and hasattr(imu_t_bs, "tensor"):
            imu_tensor = imu_t_bs.tensor()
            if imu_tensor.numel() == 0:
                R_BC = R_BC @ _EDN2NED_ROT.to(device=R_BC.device)
                return R_BC, t_BC, True

        return R_BC, t_BC, False

    def _seed_from_drt(self, drt_result, frame: T_SensorFrame) -> None:
        """Seed frontend IMU context from DRT result and push kf-0 pose to graph."""
        Logger.write("info",
            f"DRT result — scale={drt_result.scale:.6f} "
            f"|v0|={drt_result.v0.norm():.4f} m/s  "
            f"b_g={[round(x,5) for x in drt_result.b_g.tolist()]} rad/s  "
            f"|g_W|={drt_result.g_W.norm():.4f} m/s²  "
            f"g_W={[round(x,3) for x in drt_result.g_W.tolist()]}"
        )
        # Seed IMUContext (AirIO + EKF) if present
        if self.imu_context is not None:
            self.imu_context.seed_from_drt(drt_result)
            Logger.write("info", "IMUContext seeded from DRT result")
        # Initialize the pose graph with kf-0 (same as heuristic initialize but with DRT pose)
        depth0 = self.Frontend.estimate_depth(frame.stereo)
        # Prime MotionEstimator so its first update() call (in run_pair) doesn't fail.
        # The returned pose is discarded; the side effect (prev_pose = identity) is what matters.
        self.MotionEstimator.predict(frame, None, depth0.depth)
        # Use identity pose at kf-0 (DRT places kf-0 as world origin)
        est_pose = pp.identity_SE3(1, device=self.device)
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.stereo.T_BS,
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
            "K"           : frame.stereo.K,
            "baseline"    : frame.stereo.baseline,
        }))
        self.OutlierFilter.set_meta(frame.stereo)
        self.prev_keyframe = (frame, int(frame_idx.item()), depth0)

    def push_keyframe(self, frame: T_SensorFrame, est_pose: pp.LieTensor | torch.Tensor, need_interp: bool=False) -> torch.Tensor:
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.stereo.T_BS,
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
            self._handle_init(frame)
            return

        self._step_imu_context(frame)
        assert self.prev_keyframe is not None
        self.run_pair(self.prev_keyframe[0], frame)

    def _step_imu_context(self, frame: T_SensorFrame) -> None:
        """Advance AirIO + EKF one camera window using raw IMU ticks from the frame."""
        if self.imu_context is None or not hasattr(frame, "imu"):
            return
        imu = frame.imu                                          # type: ignore[attr-defined]
        dt_ns  = imu.time_delta[0, :, 0].to(torch.float64)      # (N-1,) ns
        gyro   = imu.gyro[0].to(torch.float64)                   # (N, 3)
        acc    = imu.acc[0].to(torch.float64)                    # (N, 3)
        n = dt_ns.shape[0]
        # Without AirIMU correction we pass raw readings; EKF uses its internal
        # bias estimate (b_g, b_a from DRT init) to correct them.
        corrected = [{"acc": acc[t], "gyro": gyro[t], "dt": dt_ns[t].item() * 1e-9} for t in range(n)]
        raw      = [{"acc": acc[t], "gyro": gyro[t]} for t in range(n)]
        self.imu_context.step(corrected, raw)

    def get_map(self) -> VisualMap:
        return self.graph

    def terminate(self) -> None:
        super().terminate()
        if self.prev_keyframe is not None:
            self.Optimizer.write_map(self.graph)
        self.Optimizer.terminate()
        self.MapRefiner.elaborate_map(self.graph.frames)

    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)

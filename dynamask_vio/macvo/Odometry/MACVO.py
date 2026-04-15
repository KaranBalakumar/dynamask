import torch
import pypose as pp
import typing as T
from types import SimpleNamespace

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import StereoFrame
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode, IMUFactorNode
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.Timer import Timer
from Utility.Visualize import fig_plt
from Utility.Extensions import ConfigTestable

from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=StereoFrame)


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
        dynamic_gating  = None,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")
        
        self.graph = VisualMap()
        self.device = device
        self.mapping: bool = mapping
        self.match_cov_default: float = match_cov_default
        self.dynamic_gating_enabled = bool(getattr(dynamic_gating, "enabled", False))
        self.min_static_conf = float(getattr(dynamic_gating, "min_static_conf", 0.2))
        self.gravity_w = torch.tensor([0.0, 0.0, 9.80665], dtype=torch.float32)

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
        if hasattr(config.args, "dynamic_gating") and config.args.dynamic_gating is not None:
            cls._enforce_config_spec(config.args.dynamic_gating, {
                "enabled": lambda b: isinstance(b, bool),
                "min_static_conf": lambda v: isinstance(v, (float, int)) and 0.0 <= float(v) <= 1.0,
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
            "vel_w"       : torch.zeros((1, 3), dtype=torch.float32),
            "bias_g"      : torch.zeros((1, 3), dtype=torch.float32),
            "bias_a"      : torch.zeros((1, 3), dtype=torch.float32),
        }))
        self.OutlierFilter.set_meta(frame0.stereo)
        self.prev_keyframe = (frame0, int(frame_idx.item()), depth0)

    @staticmethod
    def _estimate_velocity(prev_pose: pp.LieTensor, curr_pose: pp.LieTensor, dt_sec: float) -> torch.Tensor:
        dt = max(float(dt_sec), 1e-3)
        p_prev = pp.SE3(prev_pose).translation()
        p_curr = pp.SE3(curr_pose).translation()
        return (p_curr - p_prev) / dt

    def run_pair(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        assert self.prev_keyframe is not None
        
        # Check if current frame is the keyframe ########################################
        if not self.KeyframeSelector.isKeyframe(frame1):            
            prev_idx = self.prev_keyframe[1]
            self.push_keyframe(
                frame1,
                self.graph.frames.data["pose"][prev_idx].unsqueeze(0),
                need_interp=True,
                vel_w=self.graph.frames.data["vel_w"][prev_idx].unsqueeze(0),
                bias_g=self.graph.frames.data["bias_g"][prev_idx].unsqueeze(0),
                bias_a=self.graph.frames.data["bias_a"][prev_idx].unsqueeze(0),
            )
            return
        
        depth0          = self.prev_keyframe[2]
        if hasattr(frame0, "imu"):
            setattr(frame0.stereo, "imu", getattr(frame0, "imu"))
        if hasattr(frame1, "imu"):
            setattr(frame1.stereo, "imu", getattr(frame1, "imu"))
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

        kp_static_conf = torch.ones((kp0_uv.size(0), 1), device=self.device)
        kp_static_weight = torch.ones((kp0_uv.size(0), 1), device=self.device)

        if hasattr(match01, "static_conf") and (getattr(match01, "static_conf") is not None):
            static_conf_map = getattr(match01, "static_conf")
            if static_conf_map.ndim == 3:
                static_conf_map = static_conf_map.unsqueeze(1)
            static_conf_px = self.Frontend.retrieve_pixels(kp0_uv, static_conf_map)
            if static_conf_px is not None:
                kp_static_conf = static_conf_px.T.to(self.device)

        if hasattr(match01, "static_weight") and (getattr(match01, "static_weight") is not None):
            static_weight_map = getattr(match01, "static_weight")
            if static_weight_map.ndim == 3:
                static_weight_map = static_weight_map.unsqueeze(1)
            static_weight_px = self.Frontend.retrieve_pixels(kp0_uv, static_weight_map)
            if static_weight_px is not None:
                kp_static_weight = static_weight_px.T.to(self.device)
        else:
            kp_static_weight = kp_static_conf

        if self.dynamic_gating_enabled:
            keep = kp_static_conf.squeeze(-1) > self.min_static_conf
            kp0_uv = kp0_uv[keep]
            kp1_uv = kp1_uv[keep]
            kp_static_conf = kp_static_conf[keep]
            kp_static_weight = kp_static_weight[keep]
        
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
            "static_conf"    : kp_static_conf.cpu(),
            "static_weight"  : kp_static_weight.cpu(),
            
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
        dt_sec = (float(frame1.stereo.frame_ns) - float(frame0.stereo.frame_ns)) * 1e-9
        prev_vel = self.graph.frames.data["vel_w"][self.prev_keyframe[1]].unsqueeze(0)
        prev_bg = self.graph.frames.data["bias_g"][self.prev_keyframe[1]].unsqueeze(0)
        prev_ba = self.graph.frames.data["bias_a"][self.prev_keyframe[1]].unsqueeze(0)
        init_vel = self._estimate_velocity(prev_pose, pp.SE3(est_pose), dt_sec)

        frame_idx      = self.push_keyframe(frame1, est_pose, vel_w=init_vel, bias_g=prev_bg, bias_a=prev_ba)
        prev_frame_idx = torch.tensor([self.prev_keyframe[1]], dtype=torch.long)
        match_idx      = self.graph.match.push(match_obs)
        
        num_match_kp = len(match_obs)
        self.graph.point2match.add(point_idx, match_idx)    # Associate point -> match
        self.graph.match2point.set(match_idx, point_idx)    # Associate match -> point
        self.graph.frame2match.add(prev_frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.frame2match.add(frame_idx     , torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.match2frame1.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(prev_frame_idx.item()))    # Associate match -> frame1
        self.graph.match2frame2.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(frame_idx.item()     ))    # Associate match -> frame2

        if hasattr(match01, "imu_preintegration"):
            imu_pack = getattr(match01, "imu_preintegration")
            dt_tensor = imu_pack.get("dt", torch.ones((1, 1), dtype=torch.float32) * dt_sec).float()
            b_g_ref = prev_bg.detach().cpu().float()
            b_a_ref = prev_ba.detach().cpu().float()
            n_imu_orig = len(self.graph.imu_factors)
            imu_idx = self.graph.imu_factors.push(IMUFactorNode.init({
                "delta_R"     : imu_pack["delta_R"].float(),
                "delta_v"     : imu_pack["delta_v"].float(),
                "delta_p"     : imu_pack["delta_p"].float(),
                "Sigma_preint": imu_pack["Sigma_preint"].float(),
                "J_R_bg"      : imu_pack["J_R_bg"].float(),
                "J_v_bg"      : imu_pack["J_v_bg"].float(),
                "J_v_ba"      : imu_pack["J_v_ba"].float(),
                "J_p_bg"      : imu_pack["J_p_bg"].float(),
                "J_p_ba"      : imu_pack["J_p_ba"].float(),
                "dt"          : dt_tensor,
                "b_g_ref"     : b_g_ref,
                "b_a_ref"     : b_a_ref,
                "gravity_w"   : self.gravity_w.view(1, 3).float(),
                "sigma_bg_rw" : torch.full((1, 1), 1e-3, dtype=torch.float32),
                "sigma_ba_rw" : torch.full((1, 1), 1e-3, dtype=torch.float32),
            }))
            self.graph.imu2frame1.set(imu_idx, torch.full((imu_idx.numel(),), prev_frame_idx.item(), dtype=torch.long))
            self.graph.imu2frame2.set(imu_idx, torch.full((imu_idx.numel(),), frame_idx.item(), dtype=torch.long))
            self.graph.frame2imu.add(prev_frame_idx, torch.tensor([n_imu_orig], dtype=torch.long), torch.tensor([imu_idx.numel()], dtype=torch.long))
            self.graph.frame2imu.add(frame_idx, torch.tensor([n_imu_orig], dtype=torch.long), torch.tensor([imu_idx.numel()], dtype=torch.long))

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
                self.Optimizer.get_graph_data(self.graph, frame_idx, dynamic_gating_enabled=self.dynamic_gating_enabled)
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

    def push_keyframe(
        self,
        frame: T_SensorFrame,
        est_pose: pp.LieTensor | torch.Tensor,
        need_interp: bool = False,
        vel_w: torch.Tensor | None = None,
        bias_g: torch.Tensor | None = None,
        bias_a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vel_w = torch.zeros((1, 3), dtype=torch.float32) if vel_w is None else vel_w.float()
        bias_g = torch.zeros((1, 3), dtype=torch.float32) if bias_g is None else bias_g.float()
        bias_a = torch.zeros((1, 3), dtype=torch.float32) if bias_a is None else bias_a.float()
        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.stereo.T_BS,
            "need_interp" : torch.tensor([need_interp], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
            "K"           : frame.stereo.K,
            "baseline"    : frame.stereo.baseline,
            "vel_w"       : vel_w,
            "bias_g"      : bias_g,
            "bias_a"      : bias_a,
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
            self.isinitiated = True
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

    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)

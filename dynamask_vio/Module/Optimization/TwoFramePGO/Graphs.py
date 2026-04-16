import torch
import pypose as pp
import typing as T
from torch import nn
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph


def _resolve_observation_weight(
    obs: MatchObs,
    num_obs: int,
    device: torch.device,
    dtype: torch.dtype,
    w_eps: float,
) -> torch.Tensor:
    w = obs.data.get("static_conf", None)
    if w is None:
        return torch.ones((num_obs, 1), device=device, dtype=dtype)
    return w.reshape(-1, 1).to(device=device, dtype=dtype).clamp(min=w_eps, max=1.0)


@dataclass
class GraphInput:
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    init_motion       : pp.LieTensor
    from_pose         : pp.LieTensor | None
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str
    imu_factor        : dict[str, torch.Tensor] | None = None
    init_vel_w        : torch.Tensor | None = None
    init_bias_g       : torch.Tensor | None = None
    init_bias_a       : torch.Tensor | None = None
    from_vel_w        : torch.Tensor | None = None
    from_bias_g       : torch.Tensor | None = None
    from_bias_a       : torch.Tensor | None = None
    window_frame_idx  : torch.Tensor | None = None
    window_init_motion: pp.LieTensor | None = None
    window_init_vel_w : torch.Tensor | None = None
    window_init_bias_g: torch.Tensor | None = None
    window_init_bias_a: torch.Tensor | None = None
    window_imu_factors: dict[str, torch.Tensor] | None = None


def _ensure_imu_cov_blocks(
    factor: dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    eps = 1.0e-6
    if "Sigma_imu15" in factor:
        sigma15 = factor["Sigma_imu15"].reshape(-1, 15, 15).to(device=device, dtype=dtype)
        blocks = [
            sigma15[:, 0:3, 0:3],
            sigma15[:, 3:6, 3:6],
            sigma15[:, 6:9, 6:9],
            sigma15[:, 9:12, 9:12],
            sigma15[:, 12:15, 12:15],
        ]
    else:
        sigma = factor.get("Sigma_preint", torch.eye(9, device=device, dtype=dtype).reshape(1, 9, 9))
        sigma = sigma.reshape(-1, 9, 9).to(device=device, dtype=dtype)
        dt = factor.get("dt", torch.tensor([[0.01]], device=device, dtype=dtype)).reshape(-1, 1).to(device=device, dtype=dtype)
        bias_var = dt.clamp(min=1.0e-3) * 1.0e-3
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(sigma.shape[0], -1, -1)
        blocks = [
            sigma[:, 0:3, 0:3],
            sigma[:, 3:6, 3:6],
            sigma[:, 6:9, 6:9],
            eye * bias_var.view(-1, 1, 1),
            eye * bias_var.view(-1, 1, 1),
        ]
    cov = torch.cat(blocks, dim=0)
    eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return cov + eps * eye3


def _calc_imu_residual_blocks(
    from_pose: pp.LieTensor,
    to_pose: pp.LieTensor,
    from_vel: torch.Tensor,
    to_vel: torch.Tensor,
    from_bg: torch.Tensor,
    to_bg: torch.Tensor,
    from_ba: torch.Tensor,
    to_ba: torch.Tensor,
    delta_R: torch.Tensor,
    delta_v: torch.Tensor,
    delta_p: torch.Tensor,
    J_R_bg: torch.Tensor,
    J_v_bg: torch.Tensor,
    J_v_ba: torch.Tensor,
    J_p_bg: torch.Tensor,
    J_p_ba: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    R_i = from_pose.rotation().matrix()
    R_j = to_pose.rotation().matrix()
    p_i = from_pose.translation()
    p_j = to_pose.translation()
    dbg = from_bg - to_bg
    dba = from_ba - to_ba

    R_rel = torch.bmm(R_i.transpose(-1, -2), R_j)
    dR_corr = torch.bmm(delta_R.transpose(-1, -2), R_rel)
    r_R = pp.from_matrix(dR_corr, pp.SO3_type).Log() + torch.bmm(J_R_bg, dbg.unsqueeze(-1)).squeeze(-1)

    v_rel = torch.bmm(R_i.transpose(-1, -2), (to_vel - from_vel).unsqueeze(-1)).squeeze(-1)
    r_v = v_rel - delta_v - torch.bmm(J_v_bg, dbg.unsqueeze(-1)).squeeze(-1) - torch.bmm(J_v_ba, dba.unsqueeze(-1)).squeeze(-1)

    p_rel_world = p_j - p_i - from_vel * dt
    p_rel = torch.bmm(R_i.transpose(-1, -2), p_rel_world.unsqueeze(-1)).squeeze(-1)
    r_p = p_rel - delta_p - torch.bmm(J_p_bg, dbg.unsqueeze(-1)).squeeze(-1) - torch.bmm(J_p_ba, dba.unsqueeze(-1)).squeeze(-1)

    r_bg = to_bg - from_bg
    r_ba = to_ba - from_ba
    return torch.cat([r_R, r_v, r_p, r_bg, r_ba], dim=0)


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor
    vel_w    : torch.Tensor | None = None
    bias_g   : torch.Tensor | None = None
    bias_a   : torch.Tensor | None = None
    window_frame_idx: torch.Tensor | None = None
    window_motion   : torch.Tensor | None = None
    window_vel_w    : torch.Tensor | None = None
    window_bias_g   : torch.Tensor | None = None
    window_bias_a   : torch.Tensor | None = None


############## Optimization Graphs

class ICP_TwoframePGO(FactorGraph):
    W_EPS: T.Final[float] = 1.0e-3

    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.init_motion           = graph_data.init_motion
        self.from_idx              = graph_data.from_idx
        self.frame_idx             = graph_data.frame_idx
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index
        self.imu_cov_blocks = None
        self.imu_factor_payload = None
        self.from_pose_imu = None
        self.from_vel_imu = None
        self.from_bias_g_imu = None
        self.from_bias_a_imu = None
        self.vel2opt = None
        self.bias_g2opt = None
        self.bias_a2opt = None
        
        # ICP-based residual
        self.pts = graph_data.points
        self.obs = graph_data.observations
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.points_Tc: torch.Tensor
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])
        self.register_buffer(
            "w",
            _resolve_observation_weight(self.obs, self.points_Tc.shape[0], self.points_Tc.device, self.points_Tc.dtype, self.W_EPS),
        )
        if graph_data.imu_factor is not None and graph_data.from_pose is not None:
            def _state_or_default(value: torch.Tensor | None) -> torch.Tensor:
                if value is None:
                    return torch.zeros((1, 3), dtype=self.points_Tc.dtype, device=self.points_Tc.device)
                return value.reshape(1, 3).to(dtype=self.points_Tc.dtype, device=self.points_Tc.device)

            self.from_pose_imu = pp.SE3(graph_data.from_pose)
            self.from_vel_imu = _state_or_default(graph_data.from_vel_w).clone()
            self.from_bias_g_imu = _state_or_default(graph_data.from_bias_g).clone()
            self.from_bias_a_imu = _state_or_default(graph_data.from_bias_a).clone()
            self.vel2opt = nn.Parameter(_state_or_default(graph_data.init_vel_w).clone())
            self.bias_g2opt = nn.Parameter(_state_or_default(graph_data.init_bias_g).clone())
            self.bias_a2opt = nn.Parameter(_state_or_default(graph_data.init_bias_a).clone())
            self.imu_factor_payload = {
                "delta_R": graph_data.imu_factor["delta_R"].reshape(1, 3, 3).clone(),
                "delta_v": graph_data.imu_factor["delta_v"].reshape(1, 3).clone(),
                "delta_p": graph_data.imu_factor["delta_p"].reshape(1, 3).clone(),
                "J_R_bg": graph_data.imu_factor["J_R_bg"].reshape(1, 3, 3).clone(),
                "J_v_bg": graph_data.imu_factor["J_v_bg"].reshape(1, 3, 3).clone(),
                "J_v_ba": graph_data.imu_factor["J_v_ba"].reshape(1, 3, 3).clone(),
                "J_p_bg": graph_data.imu_factor["J_p_bg"].reshape(1, 3, 3).clone(),
                "J_p_ba": graph_data.imu_factor["J_p_ba"].reshape(1, 3, 3).clone(),
                "dt": graph_data.imu_factor["dt"].reshape(1, 1).clone(),
            }
            if "Sigma_imu15" in graph_data.imu_factor:
                self.imu_factor_payload["Sigma_imu15"] = graph_data.imu_factor["Sigma_imu15"].reshape(1, 15, 15).clone()
            else:
                self.imu_factor_payload["Sigma_preint"] = graph_data.imu_factor["Sigma_preint"].reshape(1, 9, 9).clone()
            self.imu_cov_blocks = _ensure_imu_cov_blocks(self.imu_factor_payload, self.points_Tc.dtype, self.points_Tc.device)


    def forward(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        residual_obs = self.w * (frame_pose.Act(self.points_Tc) - self.points_Tw)
        if (
            self.imu_factor_payload is None
            or self.from_pose_imu is None
            or self.vel2opt is None
            or self.bias_g2opt is None
            or self.bias_a2opt is None
        ):
            return residual_obs
        imu_payload = {k: v.to(self.vel2opt) for k, v in self.imu_factor_payload.items()}
        imu_res = _calc_imu_residual_blocks(
            self.from_pose_imu,
            pp.SE3(self.pose2opt),
            self.from_vel_imu.to(self.vel2opt),
            self.vel2opt,
            self.from_bias_g_imu.to(self.bias_g2opt),
            self.bias_g2opt,
            self.from_bias_a_imu.to(self.bias_a2opt),
            self.bias_a2opt,
            imu_payload["delta_R"],
            imu_payload["delta_v"],
            imu_payload["delta_p"],
            imu_payload["J_R_bg"],
            imu_payload["J_v_bg"],
            imu_payload["J_v_ba"],
            imu_payload["J_p_bg"],
            imu_payload["J_p_ba"],
            imu_payload["dt"],
        )
        return torch.cat([residual_obs, imu_res], dim=0)
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        cov_obs = (R @ self.obs_covTc @ RT) + self.pts_covTw # type: ignore
        if self.imu_cov_blocks is None:
            return cov_obs
        return torch.cat([cov_obs, self.imu_cov_blocks.to(cov_obs)], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        vel = None if self.vel2opt is None else self.vel2opt
        bg = None if self.bias_g2opt is None else self.bias_g2opt
        ba = None if self.bias_a2opt is None else self.bias_a2opt
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx, vel_w=vel, bias_g=bg, bias_a=ba)


class Reproj_TwoFramePGO(FactorGraph):
    W_EPS: T.Final[float] = 1.0e-3

    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.from_idx : torch.Tensor = graph_data.from_idx
        self.frame_idx: torch.Tensor = graph_data.frame_idx
        self.init_motion:  pp.LieTensor = graph_data.init_motion
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index
        
        self.pts     = graph_data.points
        self.obs     = graph_data.observations

        self.pos_Tc: torch.Tensor
        self.pos_Tw: torch.Tensor
        self.K: torch.Tensor
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("pos_Tw" , self.pts.data["pos_Tw"])
        self.register_buffer("cov_Tw" , self.pts.data["cov_Tw"])
        self.register_buffer("kp2"    , self.obs.data["pixel2_uv"])
        
        N = self.obs.data["pixel2_uv_cov"].size(0)
        cov_kp2 = torch.empty((N, 2, 2))
        cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.register_buffer("cov_kp2", cov_kp2)
        self.register_buffer(
            "w",
            _resolve_observation_weight(self.obs, self.kp2.shape[0], self.kp2.device, self.kp2.dtype, self.W_EPS),
        )

    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        residual = point2pixel_NED(self.pos_Tc, self.K) - self.kp2
        return self.w * residual

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return T.cast(torch.Tensor, self.cov_kp2)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        with torch.no_grad():
            return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class ReprojDisp_TwoFramePGO(Reproj_TwoFramePGO):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        self.register_buffer("baseline", graph_data.baseline)
        self.baseline: torch.Tensor
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])
        self.imu_factor_payload = None
        self.from_pose_imu = None
        self.from_vel_imu = None
        self.from_bias_g_imu = None
        self.from_bias_a_imu = None
        self.imu_cov_blocks = None
        self.vel2opt = None
        self.bias_g2opt = None
        self.bias_a2opt = None

        cov_kp2 = T.cast(torch.Tensor, self.cov_kp2)

        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)
        if graph_data.imu_factor is not None and graph_data.from_pose is not None:
            def _state_or_default(value: torch.Tensor | None) -> torch.Tensor:
                if value is None:
                    return torch.zeros((1, 3), dtype=self.kp2.dtype, device=self.kp2.device)
                return value.reshape(1, 3).to(dtype=self.kp2.dtype, device=self.kp2.device)

            self.from_pose_imu = pp.SE3(graph_data.from_pose)
            self.from_vel_imu = _state_or_default(graph_data.from_vel_w).clone()
            self.from_bias_g_imu = _state_or_default(graph_data.from_bias_g).clone()
            self.from_bias_a_imu = _state_or_default(graph_data.from_bias_a).clone()
            self.vel2opt = nn.Parameter(_state_or_default(graph_data.init_vel_w).clone())
            self.bias_g2opt = nn.Parameter(_state_or_default(graph_data.init_bias_g).clone())
            self.bias_a2opt = nn.Parameter(_state_or_default(graph_data.init_bias_a).clone())
            self.imu_factor_payload = {
                "delta_R": graph_data.imu_factor["delta_R"].reshape(1, 3, 3).clone(),
                "delta_v": graph_data.imu_factor["delta_v"].reshape(1, 3).clone(),
                "delta_p": graph_data.imu_factor["delta_p"].reshape(1, 3).clone(),
                "J_R_bg": graph_data.imu_factor["J_R_bg"].reshape(1, 3, 3).clone(),
                "J_v_bg": graph_data.imu_factor["J_v_bg"].reshape(1, 3, 3).clone(),
                "J_v_ba": graph_data.imu_factor["J_v_ba"].reshape(1, 3, 3).clone(),
                "J_p_bg": graph_data.imu_factor["J_p_bg"].reshape(1, 3, 3).clone(),
                "J_p_ba": graph_data.imu_factor["J_p_ba"].reshape(1, 3, 3).clone(),
                "dt": graph_data.imu_factor["dt"].reshape(1, 1).clone(),
            }
            if "Sigma_imu15" in graph_data.imu_factor:
                self.imu_factor_payload["Sigma_imu15"] = graph_data.imu_factor["Sigma_imu15"].reshape(1, 15, 15).clone()
            else:
                self.imu_factor_payload["Sigma_preint"] = graph_data.imu_factor["Sigma_preint"].reshape(1, 9, 9).clone()
            self.imu_cov_blocks = _ensure_imu_cov_blocks(self.imu_factor_payload, self.kp2.dtype, self.kp2.device)
    
    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw
        K = T.cast(torch.Tensor, self.K)
        bl = T.cast(torch.Tensor, self.baseline)

        reproj_err = point2pixel_NED(self.pos_Tc, K) - T.cast(torch.Tensor, self.kp2)
        depth_err = (self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)) - self.kp2_disparity
        residual = self.w * torch.cat((reproj_err, depth_err), dim=-1)
        if (
            self.imu_factor_payload is None
            or self.from_pose_imu is None
            or self.vel2opt is None
            or self.bias_g2opt is None
            or self.bias_a2opt is None
        ):
            return residual
        imu_payload = {k: v.to(self.vel2opt) for k, v in self.imu_factor_payload.items()}
        imu_res = _calc_imu_residual_blocks(
            self.from_pose_imu,
            pp.SE3(self.pose2opt),
            self.from_vel_imu.to(self.vel2opt),
            self.vel2opt,
            self.from_bias_g_imu.to(self.bias_g2opt),
            self.bias_g2opt,
            self.from_bias_a_imu.to(self.bias_a2opt),
            self.bias_a2opt,
            imu_payload["delta_R"],
            imu_payload["delta_v"],
            imu_payload["delta_p"],
            imu_payload["J_R_bg"],
            imu_payload["J_v_bg"],
            imu_payload["J_v_ba"],
            imu_payload["J_p_bg"],
            imu_payload["J_p_ba"],
            imu_payload["dt"],
        )
        return torch.cat([residual, imu_res], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        cov = T.cast(torch.Tensor, self.cov)
        if self.imu_cov_blocks is None:
            return cov
        return torch.cat([cov, self.imu_cov_blocks.to(cov)], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        vel = None if self.vel2opt is None else self.vel2opt
        bg = None if self.bias_g2opt is None else self.bias_g2opt
        ba = None if self.bias_a2opt is None else self.bias_a2opt
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx, vel_w=vel, bias_g=bg, bias_a=ba)


class SlidingWindow_ReprojDisp_PGO(FactorGraph):
    W_EPS: T.Final[float] = 1.0e-3

    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        if (
            graph_data.window_frame_idx is None
            or graph_data.window_init_motion is None
            or graph_data.window_init_vel_w is None
            or graph_data.window_init_bias_g is None
            or graph_data.window_init_bias_a is None
        ):
            raise ValueError("SlidingWindow_ReprojDisp_PGO requires window frame/state payload.")

        self.from_idx = graph_data.from_idx
        self.frame_idx = graph_data.frame_idx
        self.window_frame_idx = graph_data.window_frame_idx
        self.pose_window = pp.Parameter(pp.SE3(graph_data.window_init_motion))
        self.vel_window = nn.Parameter(graph_data.window_init_vel_w.clone())
        self.bias_g_window = nn.Parameter(graph_data.window_init_bias_g.clone())
        self.bias_a_window = nn.Parameter(graph_data.window_init_bias_a.clone())

        self.register_buffer("baseline", graph_data.baseline)
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("kp2", graph_data.observations.data["pixel2_uv"])
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])
        self.register_buffer("pos_Tw", graph_data.points.data["pos_Tw"])
        self.edges_index = graph_data.edges_index

        cov_kp2 = torch.empty((self.kp2.size(0), 2, 2))
        cov_kp2[:, 0, 0] = graph_data.observations.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = graph_data.observations.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = graph_data.observations.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = graph_data.observations.data["pixel2_uv_cov"][:, 2]
        cov = torch.zeros((self.kp2.size(0), 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)
        self.register_buffer(
            "w",
            _resolve_observation_weight(graph_data.observations, self.kp2.shape[0], self.kp2.device, self.kp2.dtype, self.W_EPS),
        )

        self.window_imu = graph_data.window_imu_factors
        self.imu_cov_blocks = None
        if self.window_imu is not None:
            self.imu_cov_blocks = _ensure_imu_cov_blocks(self.window_imu, self.kp2.dtype, self.kp2.device)

        self.register_buffer("anchor_pose", pp.SE3(graph_data.window_init_motion[0:1]).tensor())
        self.register_buffer("anchor_vel", graph_data.window_init_vel_w[0:1].clone())
        self.register_buffer("anchor_bg", graph_data.window_init_bias_g[0:1].clone())
        self.register_buffer("anchor_ba", graph_data.window_init_bias_a[0:1].clone())
        prior_cov = torch.eye(3).reshape(1, 3, 3) * 1.0e-4
        self.register_buffer("anchor_cov_blocks", prior_cov.repeat(5, 1, 1))

    def _visual_residual(self) -> torch.Tensor:
        pose_last = pp.SE3(self.pose_window[-1:])
        pose_obs = pose_last[self.edges_index]
        pos_Tc = pose_obs.Inv() * self.pos_Tw
        reproj_err = point2pixel_NED(pos_Tc, self.K) - self.kp2
        depth_err = (pos_Tc[:, 0:1].reciprocal() * (self.K[0, 0] * self.baseline)) - self.kp2_disparity
        return self.w * torch.cat((reproj_err, depth_err), dim=-1)

    def _imu_residual(self) -> torch.Tensor:
        if self.window_imu is None:
            return torch.empty((0, 3), device=self.pose_window.device, dtype=self.pose_window.dtype)
        residual_blocks = []
        n = self.window_frame_idx.numel()
        for i in range(n - 1):
            factor_i = {k: v[i:i + 1] for k, v in self.window_imu.items()}
            residual_blocks.append(
                _calc_imu_residual_blocks(
                    pp.SE3(self.pose_window[i:i + 1]),
                    pp.SE3(self.pose_window[i + 1:i + 2]),
                    self.vel_window[i:i + 1],
                    self.vel_window[i + 1:i + 2],
                    self.bias_g_window[i:i + 1],
                    self.bias_g_window[i + 1:i + 2],
                    self.bias_a_window[i:i + 1],
                    self.bias_a_window[i + 1:i + 2],
                    factor_i["delta_R"],
                    factor_i["delta_v"],
                    factor_i["delta_p"],
                    factor_i["J_R_bg"],
                    factor_i["J_v_bg"],
                    factor_i["J_v_ba"],
                    factor_i["J_p_bg"],
                    factor_i["J_p_ba"],
                    factor_i["dt"],
                )
            )
        return torch.cat(residual_blocks, dim=0) if residual_blocks else torch.empty((0, 3), device=self.pose_window.device, dtype=self.pose_window.dtype)

    def _anchor_residual(self) -> torch.Tensor:
        pose0 = pp.SE3(self.pose_window[0:1])
        anchor_pose = pp.SE3(self.anchor_pose)
        r_R = (anchor_pose.rotation().Inv() @ pose0.rotation()).Log()
        r_p = pose0.translation() - anchor_pose.translation()
        r_v = self.vel_window[0:1] - self.anchor_vel.to(self.vel_window)
        r_bg = self.bias_g_window[0:1] - self.anchor_bg.to(self.bias_g_window)
        r_ba = self.bias_a_window[0:1] - self.anchor_ba.to(self.bias_a_window)
        return torch.cat([r_R, r_v, r_p, r_bg, r_ba], dim=0)

    def forward(self) -> torch.Tensor:
        residual = [self._visual_residual(), self._imu_residual(), self._anchor_residual()]
        return torch.cat([r for r in residual if r.numel() > 0], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        cov_list = [self.cov]
        if self.imu_cov_blocks is not None:
            cov_list.append(self.imu_cov_blocks.to(self.cov))
        cov_list.append(self.anchor_cov_blocks.to(self.cov))
        return torch.cat(cov_list, dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(
            motion=pp.SE3(self.pose_window[-1:]),
            frame_idx=self.frame_idx,
            from_idx=self.from_idx,
            vel_w=self.vel_window[-1:],
            bias_g=self.bias_g_window[-1:],
            bias_a=self.bias_a_window[-1:],
            window_frame_idx=self.window_frame_idx,
            window_motion=pp.SE3(self.pose_window),
            window_vel_w=self.vel_window,
            window_bias_g=self.bias_g_window,
            window_bias_a=self.bias_a_window,
        )


class Analytic_ICP_TwoframePGO(ICP_TwoframePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R = frame_pose.rotation().matrix()
        p = self.points_Tc
        E = p.shape[0]
        param_cols = 7 + (0 if self.vel2opt is None else 9)

        J = torch.zeros((E, 3, 7), device=p.device, dtype=p.dtype)

        I3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J[..., 0:3] = I3
        J[..., 3:6] = -pp.vec2skew(frame_pose.Act(p))
        J_obs_pose = (J * self.w.view(E, 1, 1)).view(-1, 7)
        if self.vel2opt is None or self.bias_g2opt is None or self.bias_a2opt is None:
            return J_obs_pose

        J_obs = torch.zeros((J_obs_pose.shape[0], param_cols), device=p.device, dtype=p.dtype)
        J_obs[:, :7] = J_obs_pose
        J_imu = torch.zeros((5, 3, param_cols), device=p.device, dtype=p.dtype)
        eye3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J_imu[0, :, 3:6] = eye3
        J_imu[1, :, 7:10] = eye3
        J_imu[1, :, 10:13] = -eye3
        J_imu[1, :, 13:16] = -eye3
        J_imu[2, :, 0:3] = eye3
        J_imu[2, :, 7:10] = -eye3 * self.imu_factor_payload["dt"].reshape(1, 1, 1).to(dtype=p.dtype)
        J_imu[2, :, 10:13] = -eye3
        J_imu[2, :, 13:16] = -eye3
        J_imu[3, :, 10:13] = eye3
        J_imu[4, :, 13:16] = eye3
        return torch.cat([J_obs, J_imu.view(-1, param_cols)], dim=0)


class Analytic_Reproj_TwoFramePGO(Reproj_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x

        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J = (J_homoKS @ J_Tinv_p)
        J = J * self.w.view(self.pos_Tc.shape[0], 1, 1)
        J = J.view(-1, 7)
        return J


class Analytic_ReprojDisp_TwoFramePGO(ReprojDisp_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x
        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J_reproj = (J_homoKS @ J_Tinv_p)
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]
        J = torch.cat((J_reproj, J_disp), dim=1)
        J = J * self.w.view(self.pos_Tc.shape[0], 1, 1)
        J_obs_pose = J.view(-1, 7)
        if self.vel2opt is None or self.bias_g2opt is None or self.bias_a2opt is None:
            return J_obs_pose

        param_cols = 16
        J_obs = torch.zeros((J_obs_pose.shape[0], param_cols), device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_obs[:, :7] = J_obs_pose
        J_imu = torch.zeros((5, 3, param_cols), device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        eye3 = torch.eye(3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype).unsqueeze(0)
        J_imu[0, :, 3:6] = eye3
        J_imu[1, :, 7:10] = eye3
        J_imu[1, :, 10:13] = -eye3
        J_imu[1, :, 13:16] = -eye3
        J_imu[2, :, 0:3] = eye3
        J_imu[2, :, 7:10] = -eye3 * self.imu_factor_payload["dt"].reshape(1, 1, 1).to(dtype=self.pos_Tc.dtype)
        J_imu[2, :, 10:13] = -eye3
        J_imu[2, :, 13:16] = -eye3
        J_imu[3, :, 10:13] = eye3
        J_imu[4, :, 13:16] = eye3
        return torch.cat([J_obs, J_imu.view(-1, param_cols)], dim=0)

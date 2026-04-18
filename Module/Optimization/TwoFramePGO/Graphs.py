import torch
import pypose as pp
import typing as T
import torch.nn as nn
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode, IMUEdgeNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from Module.Network.AirIMU.preintegration import so3_exp, so3_log
from ..PyposeOptimizers import AnalyticModule, CovarianceArray, FactorGraph


@dataclass
class GraphInput:
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    init_motion       : pp.LieTensor
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str
    from_pose         : torch.Tensor | None = None
    T_BS              : torch.Tensor | None = None
    imu_edge          : IMUEdgeNode | None = None
    init_v_i          : torch.Tensor | None = None
    init_v_j          : torch.Tensor | None = None
    init_bg_i         : torch.Tensor | None = None
    init_bg_j         : torch.Tensor | None = None
    init_ba_i         : torch.Tensor | None = None
    init_ba_j         : torch.Tensor | None = None
    gravity           : torch.Tensor | None = None


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor
    vel      : torch.Tensor | None = None
    bias_g   : torch.Tensor | None = None
    bias_a   : torch.Tensor | None = None
    diagnostics: dict[str, float] | None = None
    dump_payload: dict[str, torch.Tensor] | None = None


############## Optimization Graphs

class ICP_TwoframePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.init_motion           = graph_data.init_motion
        self.from_idx              = graph_data.from_idx
        self.frame_idx             = graph_data.frame_idx
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index
        
        # ICP-based residual
        self.pts = graph_data.points
        self.obs = graph_data.observations

        self.K: torch.Tensor
        self.points_Tc: torch.Tensor
        self.points_Tw: torch.Tensor
        self.obs_covTc: torch.Tensor
        self.pts_covTw: torch.Tensor
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])
        

    def forward(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        return frame_pose.Act(self.points_Tc) - self.points_Tw
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        return (R @ self.obs_covTc @ RT) + self.pts_covTw

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx)


class Reproj_TwoFramePGO(FactorGraph):
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
        self.cov_Tw: torch.Tensor
        self.K: torch.Tensor
        self.kp2: torch.Tensor
        self.cov_kp2: torch.Tensor
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

    def forward(self) -> torch.Tensor:
        pose2opt = T.cast(pp.LieTensor, self.pose2opt)
        self.pos_Tc = pose2opt.Inv().Act(self.pos_Tw)
        return point2pixel_NED(self.pos_Tc, self.K) - self.kp2

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> CovarianceArray:
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
        self.kp2_disparity: torch.Tensor
        self.cov: torch.Tensor

        cov_kp2 = T.cast(torch.Tensor, self.cov_kp2)

        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)
    
    def forward(self) -> torch.Tensor:
        pose2opt = T.cast(pp.LieTensor, self.pose2opt)
        self.pos_Tc = pose2opt.Inv() * self.pos_Tw
        K = T.cast(torch.Tensor, self.K)
        bl = T.cast(torch.Tensor, self.baseline)

        reproj_err = point2pixel_NED(self.pos_Tc, K) - self.kp2
        depth_err = (self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)) - self.kp2_disparity
        return torch.cat((reproj_err, depth_err), dim=-1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return T.cast(torch.Tensor, self.cov)


class Reproj_TwoFramePGO_IMU(Reproj_TwoFramePGO):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        assert graph_data.imu_edge is not None, "IMU edge required for reproj_imu graph."
        assert graph_data.from_pose is not None and graph_data.T_BS is not None
        assert graph_data.init_v_i is not None and graph_data.init_v_j is not None
        assert graph_data.init_bg_i is not None and graph_data.init_bg_j is not None
        assert graph_data.init_ba_i is not None and graph_data.init_ba_j is not None
        assert graph_data.gravity is not None

        self.v_j: nn.Parameter = nn.Parameter(graph_data.init_v_j.double())
        self.b_gj: nn.Parameter = nn.Parameter(graph_data.init_bg_j.double())
        self.b_aj: nn.Parameter = nn.Parameter(graph_data.init_ba_j.double())

        self.v_i: torch.Tensor
        self.b_gi: torch.Tensor
        self.b_ai: torch.Tensor
        self.g_W: torch.Tensor
        self.R_i: torch.Tensor
        self.p_i: torch.Tensor
        self.T_BS: torch.Tensor
        self.imu_delta_R: torch.Tensor
        self.imu_delta_v: torch.Tensor
        self.imu_delta_p: torch.Tensor
        self.imu_Sigma: torch.Tensor
        self.imu_dt: torch.Tensor
        self.imu_bias_ref: torch.Tensor
        self.imu_J_R_bg: torch.Tensor
        self.imu_J_v_bg: torch.Tensor
        self.imu_J_v_ba: torch.Tensor
        self.imu_J_p_bg: torch.Tensor
        self.imu_J_p_ba: torch.Tensor
        self.register_buffer("v_i", graph_data.init_v_i.double())
        self.register_buffer("b_gi", graph_data.init_bg_i.double())
        self.register_buffer("b_ai", graph_data.init_ba_i.double())
        self.register_buffer("g_W", graph_data.gravity.double())

        T_BS = T.cast(pp.LieTensor, pp.SE3(graph_data.T_BS).double())
        T_WC_i = T.cast(pp.LieTensor, pp.SE3(graph_data.from_pose).double())
        T_WB_i = T.cast(pp.LieTensor, T_WC_i @ T_BS.Inv())
        self.register_buffer("R_i", T_WB_i.rotation().matrix()[0])
        self.register_buffer("p_i", T_WB_i.translation()[0])
        self.register_buffer("T_BS", graph_data.T_BS.double())

        e = graph_data.imu_edge
        self.register_buffer("imu_delta_R", e.data["delta_R"][0].double())
        self.register_buffer("imu_delta_v", e.data["delta_v"][0].double())
        self.register_buffer("imu_delta_p", e.data["delta_p"][0].double())
        self.register_buffer("imu_Sigma", e.data["Sigma"][0].double())
        self.register_buffer("imu_dt", e.data["dt"][0].double())
        self.register_buffer("imu_bias_ref", e.data["bias_ref"][0].double())
        self.register_buffer("imu_J_R_bg", e.data["J_R_bg"][0].double())
        self.register_buffer("imu_J_v_bg", e.data["J_v_bg"][0].double())
        self.register_buffer("imu_J_v_ba", e.data["J_v_ba"][0].double())
        self.register_buffer("imu_J_p_bg", e.data["J_p_bg"][0].double())
        self.register_buffer("imu_J_p_ba", e.data["J_p_ba"][0].double())

    def _compute_imu_residual(self) -> torch.Tensor:
        pose2opt = T.cast(pp.LieTensor, self.pose2opt)
        T_WC_j = T.cast(pp.LieTensor, pp.SE3(pose2opt))
        T_BS = T.cast(pp.LieTensor, pp.SE3(self.T_BS))
        T_WB_j = T.cast(pp.LieTensor, T_WC_j @ T_BS.Inv())
        R_j = T_WB_j.rotation().matrix()[0]
        p_j = T_WB_j.translation()[0]

        dbg = self.b_gi - self.imu_bias_ref[:3]
        dba = self.b_ai - self.imu_bias_ref[3:]
        R_i_T = self.R_i.transpose(-1, -2)
        imu_dt = self.imu_dt

        dR_corr = self.imu_delta_R @ so3_exp((self.imu_J_R_bg @ dbg).unsqueeze(0))[0]
        dv_corr = self.imu_delta_v + self.imu_J_v_bg @ dbg + self.imu_J_v_ba @ dba
        dp_corr = self.imu_delta_p + self.imu_J_p_bg @ dbg + self.imu_J_p_ba @ dba

        r_R = so3_log((dR_corr.transpose(-1, -2) @ (R_i_T @ R_j)).unsqueeze(0))[0]
        r_v = R_i_T @ (self.v_j - self.v_i - self.g_W * imu_dt) - dv_corr
        r_p = R_i_T @ (p_j - self.p_i - self.v_i * imu_dt - 0.5 * self.g_W * (imu_dt ** 2)) - dp_corr
        r_bg = self.b_gj - self.b_gi
        r_ba = self.b_aj - self.b_ai
        return torch.cat([r_R, r_v, r_p, r_bg, r_ba], dim=0)

    def forward(self) -> torch.Tensor:
        vis = super().forward().reshape(-1)
        r_imu = self._compute_imu_residual()
        return torch.cat([vis, r_imu], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> list[torch.Tensor]:
        vis_cov = super().covariance_array()
        dt = torch.clamp(self.imu_dt, min=1e-5)
        sigma_bgw2 = 1e-4
        sigma_baw2 = 1e-3
        imu_cov = torch.zeros((15, 15), dtype=self.imu_Sigma.dtype, device=self.imu_Sigma.device)
        imu_cov[:9, :9] = self.imu_Sigma
        imu_cov[9:12, 9:12] = torch.eye(3, dtype=imu_cov.dtype, device=imu_cov.device) * (sigma_bgw2 * dt)
        imu_cov[12:15, 12:15] = torch.eye(3, dtype=imu_cov.dtype, device=imu_cov.device) * (sigma_baw2 * dt)
        if isinstance(vis_cov, list):
            return [*vis_cov, imu_cov]
        return [*torch.unbind(vis_cov, dim=0), imu_cov]

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(
            motion=self.pose2opt,
            frame_idx=self.frame_idx,
            from_idx=self.from_idx,
            vel=self.v_j.detach(),
            bias_g=self.b_gj.detach(),
            bias_a=self.b_aj.detach(),
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

        J = torch.zeros((E, 3, 7), device=p.device, dtype=p.dtype)

        I3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J[..., 0:3] = I3
        J[..., 3:6] = -pp.vec2skew(frame_pose.Act(p))

        return J.view(-1, 7)


class Analytic_Reproj_TwoFramePGO(Reproj_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x

        pose2opt = T.cast(pp.LieTensor, self.pose2opt)
        R = pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J = (J_homoKS @ J_Tinv_p).view(-1, 7)
        return J


class Analytic_ReprojDisp_TwoFramePGO(ReprojDisp_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x
        pose2opt = T.cast(pp.LieTensor, self.pose2opt)
        R = pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J_reproj = (J_homoKS @ J_Tinv_p)
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]
        J = torch.cat((J_reproj, J_disp), dim=1).view(-1, 7)
        return J

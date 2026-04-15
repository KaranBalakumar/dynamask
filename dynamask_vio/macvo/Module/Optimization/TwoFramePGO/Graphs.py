import torch
import pypose as pp
import typing as T
from dataclasses import dataclass

from Module.Map import FrameNode, IMUFactorNode, MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from ..PyposeOptimizers import AnalyticModule, FactorGraph


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
    dynamic_gating_enabled: bool = False
    frame_from        : FrameNode | None = None
    frame_to          : FrameNode | None = None
    imu_factors       : IMUFactorNode | None = None


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor
    vel_w    : torch.Tensor | None = None
    bias_g   : torch.Tensor | None = None
    bias_a   : torch.Tensor | None = None


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
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.points_Tc: torch.Tensor
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])
        if graph_data.dynamic_gating_enabled:
            raise NotImplementedError(
                "Dynamic gating is only implemented for reproj/reproj_imu graph_type. "
                "Use graph_type='reproj' or 'reproj_imu' when odometry.dynamic_gating.enabled=true."
            )
        

    def forward(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        return frame_pose.Act(self.points_Tc) - self.points_Tw
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        return (R @ self.obs_covTc @ RT) + self.pts_covTw # type: ignore

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
        if "static_weight" in self.obs.data:
            w = self.obs.data["static_weight"]
        elif "static_conf" in self.obs.data:
            w = self.obs.data["static_conf"]
        else:
            w = torch.ones((N, 1), dtype=torch.float32)
        if w.ndim == 1:
            w = w.unsqueeze(-1)
        self.register_buffer("w", w.clamp(min=1e-3, max=1.0))

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
        if graph_data.dynamic_gating_enabled:
            raise NotImplementedError(
                "Dynamic gating is only implemented for reproj/reproj_imu graph_type. "
                "Use graph_type='reproj' or 'reproj_imu' when odometry.dynamic_gating.enabled=true."
            )
        self.register_buffer("baseline", graph_data.baseline)
        self.baseline: torch.Tensor
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])

        cov_kp2 = T.cast(torch.Tensor, self.cov_kp2)

        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)
    
    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw
        K = T.cast(torch.Tensor, self.K)
        bl = T.cast(torch.Tensor, self.baseline)

        reproj_err = point2pixel_NED(self.pos_Tc, K) - T.cast(torch.Tensor, self.kp2)
        depth_err = (self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)) - self.kp2_disparity
        return self.w * torch.cat((reproj_err, depth_err), dim=-1)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return T.cast(torch.Tensor, self.cov)


class ReprojIMU_TwoFramePGO(FactorGraph):
    W_EPS: float = 1e-3

    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        if graph_data.frame_from is None or graph_data.frame_to is None:
            raise RuntimeError("ReprojIMU_TwoFramePGO requires frame_from and frame_to in GraphInput.")

        self.from_idx = graph_data.from_idx
        self.frame_idx = graph_data.frame_idx
        self.edges_index = graph_data.edges_index

        self.obs = graph_data.observations
        self.pts = graph_data.points
        self.imu_factors = graph_data.imu_factors

        self.pose2opt = pp.Parameter(pp.SE3(graph_data.init_motion))
        self.vel2opt = pp.Parameter(graph_data.frame_to.data["vel_w"].clone())
        self.bg2opt = pp.Parameter(graph_data.frame_to.data["bias_g"].clone())
        self.ba2opt = pp.Parameter(graph_data.frame_to.data["bias_a"].clone())

        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("pos_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("kp2", self.obs.data["pixel2_uv"])

        n = self.obs.data["pixel2_uv_cov"].size(0)
        cov_kp2 = torch.empty((n, 2, 2))
        cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.register_buffer("cov_kp2", cov_kp2)

        if "static_weight" in self.obs.data:
            w = self.obs.data["static_weight"]
        elif "static_conf" in self.obs.data:
            w = self.obs.data["static_conf"]
        else:
            w = torch.ones((n, 1), dtype=torch.float32)
        if w.ndim == 1:
            w = w.unsqueeze(-1)
        self.register_buffer("w", w.clamp(min=self.W_EPS, max=1.0))

        self.register_buffer("prev_pose_data", graph_data.frame_from.data["pose"])
        self.register_buffer("prev_vel", graph_data.frame_from.data["vel_w"])
        self.register_buffer("prev_bg", graph_data.frame_from.data["bias_g"])
        self.register_buffer("prev_ba", graph_data.frame_from.data["bias_a"])

    def _imu_residual(self) -> torch.Tensor:
        if self.imu_factors is None or len(self.imu_factors) == 0:
            return torch.zeros((0, 15), device=self.K.device, dtype=self.K.dtype)

        dt = self.imu_factors.data["dt"][:, 0:1]
        delta_R_hat = self.imu_factors.data["delta_R"]
        delta_v_hat = self.imu_factors.data["delta_v"]
        delta_p_hat = self.imu_factors.data["delta_p"]
        J_R_bg = self.imu_factors.data["J_R_bg"]
        J_v_bg = self.imu_factors.data["J_v_bg"]
        J_v_ba = self.imu_factors.data["J_v_ba"]
        J_p_bg = self.imu_factors.data["J_p_bg"]
        J_p_ba = self.imu_factors.data["J_p_ba"]
        b_g_ref = self.imu_factors.data["b_g_ref"]
        b_a_ref = self.imu_factors.data["b_a_ref"]
        gravity = self.imu_factors.data["gravity_w"]

        pose_i = pp.SE3(self.prev_pose_data)
        pose_j = pp.SE3(self.pose2opt)
        R_i = pose_i.rotation().matrix()
        R_j = pose_j.rotation().matrix()
        p_i = pose_i.translation()
        p_j = pose_j.translation()

        v_i = self.prev_vel
        v_j = self.vel2opt
        bg_i = self.prev_bg
        ba_i = self.prev_ba
        bg_j = self.bg2opt
        ba_j = self.ba2opt

        dbg = bg_i - b_g_ref
        dba = ba_i - b_a_ref

        dR_hat = pp.mat2SO3(delta_R_hat)
        dR_corr = dR_hat * pp.so3((J_R_bg @ dbg.unsqueeze(-1)).squeeze(-1)).Exp()
        R_ij = pp.mat2SO3(torch.bmm(R_i.transpose(-1, -2), R_j))
        r_R = (dR_corr.Inv() * R_ij).Log().tensor()

        R_i_T = R_i.transpose(-1, -2)
        r_v = (
            torch.bmm(R_i_T, (v_j - v_i - gravity * dt).unsqueeze(-1)).squeeze(-1)
            - (delta_v_hat + (J_v_bg @ dbg.unsqueeze(-1)).squeeze(-1) + (J_v_ba @ dba.unsqueeze(-1)).squeeze(-1))
        )
        r_p = (
            torch.bmm(
                R_i_T,
                (p_j - p_i - v_i * dt - 0.5 * gravity * (dt**2)).unsqueeze(-1),
            ).squeeze(-1)
            - (delta_p_hat + (J_p_bg @ dbg.unsqueeze(-1)).squeeze(-1) + (J_p_ba @ dba.unsqueeze(-1)).squeeze(-1))
        )
        r_bg = bg_j - bg_i
        r_ba = ba_j - ba_i
        return torch.cat([r_R, r_v, r_p, r_bg, r_ba], dim=-1)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        vis = self.w * (point2pixel_NED(pos_Tc, self.K) - self.kp2)
        imu = self._imu_residual()
        return vis, imu

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> list[torch.Tensor]:
        blocks: list[torch.Tensor] = [self.cov_kp2[i] for i in range(self.cov_kp2.size(0))]
        if self.imu_factors is not None and len(self.imu_factors) > 0:
            sigma_pre = self.imu_factors.data["Sigma_preint"]
            sigma_bg = self.imu_factors.data["sigma_bg_rw"][:, 0]
            sigma_ba = self.imu_factors.data["sigma_ba_rw"][:, 0]
            for i in range(sigma_pre.size(0)):
                cov_imu = torch.zeros((15, 15), dtype=sigma_pre.dtype, device=sigma_pre.device)
                cov_imu[:9, :9] = sigma_pre[i]
                cov_imu[9:12, 9:12] = torch.eye(3, device=sigma_pre.device, dtype=sigma_pre.dtype) * (sigma_bg[i] ** 2)
                cov_imu[12:15, 12:15] = torch.eye(3, device=sigma_pre.device, dtype=sigma_pre.dtype) * (sigma_ba[i] ** 2)
                blocks.append(cov_imu)
        return blocks

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(
            motion=self.pose2opt,
            frame_idx=self.frame_idx,
            from_idx=self.from_idx,
            vel_w=self.vel2opt.detach(),
            bias_g=self.bg2opt.detach(),
            bias_a=self.ba2opt.detach(),
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

        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J = (J_homoKS @ J_Tinv_p).view(-1, 7)
        J = J * self.w.expand(-1, 2).reshape(-1, 1)
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
        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J_reproj = (J_homoKS @ J_Tinv_p)
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]
        J = torch.cat((J_reproj, J_disp), dim=1).view(-1, 7)
        J = J * self.w.expand(-1, 3).reshape(-1, 1)
        return J

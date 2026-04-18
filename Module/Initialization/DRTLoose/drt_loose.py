from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import torch
import pypose as pp

from DataLoader import StereoInertialFrame
from Module.Frontend.Frontend import IFrontend
from Module.Network.AirIMU.contracts import narrow_corrector_output
from Module.Network.AirIMU.encoder import IMUEncoder
from Utility.PrettyPrint import Logger

from .gyro_bias_solver import solve_gyro_bias_lbfgs
from .linear_alignment import solve_linear_alignment
from .gravity_refine import refine_gravity_on_sphere


@dataclass
class DRTInitIMUEdge:
    delta_R: torch.Tensor
    delta_v: torch.Tensor
    delta_p: torch.Tensor
    Sigma: torch.Tensor
    dt: torch.Tensor
    bias_ref: torch.Tensor
    J_R_bg: torch.Tensor
    J_v_bg: torch.Tensor
    J_v_ba: torch.Tensor
    J_p_bg: torch.Tensor
    J_p_ba: torch.Tensor


@dataclass
class DRTLooseOutput:
    ok: bool
    reason: str
    rotation: list[torch.Tensor]
    position: list[torch.Tensor]
    velocity: list[torch.Tensor]
    bias_g: torch.Tensor
    bias_a: torch.Tensor
    gravity: torch.Tensor
    imu_pres: list[DRTInitIMUEdge]
    diagnostics: dict[str, Any] | None = None


def _grid_points(H: int, W: int, stride: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(stride // 2, H, stride, device=device)
    xs = torch.arange(stride // 2, W, stride, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()


def _bearing_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    v = torch.stack([x, y, torch.ones_like(x)], dim=1)
    return v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp(min=1e-12)


def _point_from_uv_depth(uv: torch.Tensor, d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * d
    y = (uv[:, 1] - cy) / fy * d
    return torch.stack([x, y, d], dim=1)


def _estimate_transform_3d3d(X0: torch.Tensor, X1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu0 = X0.mean(dim=0)
    mu1 = X1.mean(dim=0)
    X0c = X0 - mu0
    X1c = X1 - mu1
    H = X0c.transpose(0, 1) @ X1c
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    t = mu1 - R @ mu0
    return R, t


class DRTLooseInitializer:
    def __init__(
        self,
        frontend: IFrontend,
        imu_encoder: IMUEncoder,
        cfg: SimpleNamespace | None = None,
    ) -> None:
        self.frontend = frontend
        self.imu_encoder = imu_encoder
        self.cfg = cfg if cfg is not None else SimpleNamespace()

    @staticmethod
    def _imu_sample_count(frame: StereoInertialFrame) -> int:
        imu = frame.imu
        if imu.time_ns.ndim < 2 or imu.time_ns.shape[0] == 0:
            return 0
        return int(imu.time_ns.shape[1])

    def _pair_measurements(
        self,
        frame_i: StereoInertialFrame,
        frame_j: StereoInertialFrame,
        depth_i: Any,
        depth_j: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        imu_samples = self._imu_sample_count(frame_j)
        if imu_samples < 2:
            raise RuntimeError(f"Insufficient IMU samples for pair preintegration: {imu_samples}")
        _, match_ij = self.frontend.estimate_pair(frame_i.stereo, frame_j.stereo)

        flow = match_ij.flow[0].permute(1, 2, 0)  # [H,W,2]
        H, W = flow.shape[:2]
        stride = int(getattr(self.cfg, "track_stride", 16))
        uv_i = _grid_points(H, W, stride, flow.device)
        flow_samples = flow[uv_i[:, 1].long(), uv_i[:, 0].long()]
        uv_j = uv_i + flow_samples

        inb = (
            (uv_j[:, 0] >= 1)
            & (uv_j[:, 0] < (W - 1))
            & (uv_j[:, 1] >= 1)
            & (uv_j[:, 1] < (H - 1))
        )
        uv_i = uv_i[inb]
        uv_j = uv_j[inb]
        if uv_i.shape[0] < int(getattr(self.cfg, "min_pair_tracks", 40)):
            raise RuntimeError("Not enough tracked points for DRT pair")

        K = frame_i.stereo.frame_K.to(flow.device)
        bearings_i = _bearing_from_uv(uv_i, K)
        bearings_j = _bearing_from_uv(uv_j, K)

        d_i = IFrontend.retrieve_pixels(uv_i, depth_i.depth).squeeze(0)
        d_j = IFrontend.retrieve_pixels(uv_j, depth_j.depth).squeeze(0)
        valid_d = torch.isfinite(d_i) & torch.isfinite(d_j) & (d_i > 0.05) & (d_j > 0.05)
        uv_i, uv_j = uv_i[valid_d], uv_j[valid_d]
        bearings_i, bearings_j = bearings_i[valid_d], bearings_j[valid_d]
        d_i, d_j = d_i[valid_d], d_j[valid_d]
        if uv_i.shape[0] < int(getattr(self.cfg, "min_pair_tracks_depth", 12)):
            raise RuntimeError("Not enough valid depth-supported tracks for DRT pair")

        X_i = _point_from_uv_depth(uv_i, d_i, K)
        X_j = _point_from_uv_depth(uv_j, d_j, K)
        R_ij, t_ij = _estimate_transform_3d3d(X_i, X_j)
        disp_i = -(R_ij.transpose(0, 1) @ t_ij)  # displacement in frame-i camera coords

        dt = frame_j.imu.time_delta.float().squeeze(-1) * 1e-9
        seg = {
            "gyro": frame_j.imu.gyro.squeeze(0).to(flow.device),
            "dt": dt.squeeze(0).to(flow.device),
        }
        parallax = torch.linalg.vector_norm(uv_j - uv_i, dim=1)
        return bearings_i, bearings_j, disp_i, seg, parallax

    def run(self, frames: list[StereoInertialFrame], depths: list[Any], logger: Any | None = None, log_step: int | None = None) -> DRTLooseOutput:
        K = len(frames)
        diagnostics: dict[str, Any] = {"window_len": int(K)}
        if K < 3:
            return DRTLooseOutput(False, "Need >=3 init frames", [], [], [], torch.zeros(3), torch.zeros(3), torch.zeros(3), [], diagnostics)
        if not all(hasattr(f, "imu") for f in frames):
            return DRTLooseOutput(False, "Frames do not carry IMU data", [], [], [], torch.zeros(3), torch.zeros(3), torch.zeros(3), [], diagnostics)

        device = frames[0].stereo.K.device
        dtype = frames[0].stereo.K.dtype
        T_BS = cast(pp.LieTensor, pp.SE3(frames[0].stereo.T_BS).to(device=device))
        R_bc = T_BS.rotation().matrix()[0].to(dtype=dtype)
        p_bs = T_BS.translation()[0].to(dtype=dtype)

        pair_bearings: list[tuple[torch.Tensor, torch.Tensor]] = []
        pair_segments: list[dict[str, torch.Tensor]] = []
        cam_disp: list[torch.Tensor] = []
        pair_parallax: list[torch.Tensor] = []
        imu_pre_list: list[DRTInitIMUEdge] = []

        bg0 = torch.zeros((1, 3), device=device, dtype=dtype)
        ba0 = torch.zeros((1, 3), device=device, dtype=dtype)
        bias_ref0 = torch.cat([bg0, ba0], dim=1)

        for i in range(K - 1):
            try:
                b_i, b_j, d_cam, seg, parallax = self._pair_measurements(frames[i], frames[i + 1], depths[i], depths[i + 1])
            except Exception as exc:
                diagnostics["reject_reason"] = f"pair {i} failed: {exc}"
                return DRTLooseOutput(False, f"pair {i} failed: {exc}", [], [], [], torch.zeros(3), torch.zeros(3), torch.zeros(3), [], diagnostics)
            pair_bearings.append((b_i, b_j))
            pair_segments.append(seg)
            cam_disp.append(d_cam)
            pair_parallax.append(parallax)

        all_parallax = torch.cat(pair_parallax, dim=0) if len(pair_parallax) > 0 else torch.zeros((0,), dtype=torch.float32)
        diagnostics["n_views_total"] = int(sum(int(b[0].shape[0]) for b in pair_bearings))
        diagnostics["parallax_p50"] = float(all_parallax.median().item()) if all_parallax.numel() > 0 else 0.0
        diagnostics["parallax_p5"] = float(torch.quantile(all_parallax, 0.05).item()) if all_parallax.numel() > 0 else 0.0

        with torch.enable_grad():
            bg_res = solve_gyro_bias_lbfgs(
                pair_bearings=pair_bearings,
                imu_segments=pair_segments,
                R_bc=R_bc,
                init_bg=bg0[0],
                max_iter=int(getattr(self.cfg, "gyro_max_iter", 200)),
                cauchy_delta=float(getattr(self.cfg, "gyro_cauchy_delta", 1e-5)),
            )
        if not bg_res.success:
            diagnostics["reject_reason"] = f"Gyro bias solve failed: {bg_res.message}"
            diagnostics["bg_solver_residual"] = float(bg_res.final_loss)
            return DRTLooseOutput(False, f"Gyro bias solve failed: {bg_res.message}", [], [], [], torch.zeros(3), torch.zeros(3), torch.zeros(3), [], diagnostics)
        diagnostics["bg_solver_residual"] = float(bg_res.final_loss)

        bias_ref = torch.cat([bg_res.bias_g.view(1, 3), ba0], dim=1)
        for i in range(K - 1):
            imu = frames[i + 1].imu
            imu_samples = self._imu_sample_count(frames[i + 1])
            if imu_samples < 2:
                diagnostics["reject_reason"] = f"pair {i} has insufficient IMU samples: {imu_samples}"
                return DRTLooseOutput(
                    False,
                    diagnostics["reject_reason"],
                    [],
                    [],
                    [],
                    bg_res.bias_g.cpu(),
                    torch.zeros(3),
                    torch.zeros(3),
                    [],
                    diagnostics,
                )
            corr = narrow_corrector_output(self.imu_encoder.corrector.inference({"acc": imu.acc, "gyro": imu.gyro}))
            pre = self.imu_encoder.preint(
                corrected_acc=imu.acc + corr["correction_acc"],
                corrected_gyro=imu.gyro + corr["correction_gyro"],
                acc_cov=corr["cov_state"]["acc_cov"],
                gyro_cov=corr["cov_state"]["gyro_cov"],
                dt=imu.time_delta.float() * 1e-9,
                bias_ref=bias_ref.to(imu.acc.device, imu.acc.dtype),
                emit_jacobians=True,
                logger=logger,
                log_step=log_step,
            )
            imu_pre_list.append(
                DRTInitIMUEdge(
                    delta_R=pre.delta_R[0].double().cpu(),
                    delta_v=pre.delta_v[0].double().cpu(),
                    delta_p=pre.delta_p[0].double().cpu(),
                    Sigma=pre.Sigma[0].double().cpu(),
                    dt=pre.dt_total[0].double().cpu(),
                    bias_ref=pre.bias_ref[0].double().cpu(),
                    J_R_bg=pre.J_R_bg[0].double().cpu() if pre.J_R_bg is not None else torch.zeros((3, 3), dtype=torch.float64),
                    J_v_bg=pre.J_v_bg[0].double().cpu() if pre.J_v_bg is not None else torch.zeros((3, 3), dtype=torch.float64),
                    J_v_ba=pre.J_v_ba[0].double().cpu() if pre.J_v_ba is not None else torch.zeros((3, 3), dtype=torch.float64),
                    J_p_bg=pre.J_p_bg[0].double().cpu() if pre.J_p_bg is not None else torch.zeros((3, 3), dtype=torch.float64),
                    J_p_ba=pre.J_p_ba[0].double().cpu() if pre.J_p_ba is not None else torch.zeros((3, 3), dtype=torch.float64),
                )
            )

        R_body = [torch.eye(3, dtype=torch.float64)]
        for k in range(K - 1):
            R_next = R_body[-1] @ imu_pre_list[k].delta_R
            R_body.append(R_next)

        R_bc64 = R_bc.double()
        p_cam = [torch.zeros(3, dtype=torch.float64)]
        for k in range(K - 1):
            d_cam = cam_disp[k].double().cpu()
            R_cam_k = R_body[k] @ R_bc64
            p_cam.append(p_cam[-1] + R_cam_k @ d_cam)

        p_bs64 = p_bs.double().cpu()
        p_body = [p_cam[k] - R_body[k] @ p_bs64 for k in range(K)]
        d_v = torch.stack([imu_pre_list[k].delta_v for k in range(K - 1)], dim=0)
        d_p = torch.stack([imu_pre_list[k].delta_p for k in range(K - 1)], dim=0)
        dt = torch.stack([imu_pre_list[k].dt for k in range(K - 1)], dim=0)
        R_stack = torch.stack(R_body, dim=0)
        p_stack = torch.stack(p_body, dim=0)

        la = solve_linear_alignment(
            rotation_body=R_stack,
            position_body=p_stack,
            delta_v=d_v,
            delta_p=d_p,
            dt=dt,
            g_mag=float(getattr(self.cfg, "g_mag", 9.81)),
        )
        if not la.success:
            diagnostics["reject_reason"] = f"Linear alignment failed: {la.message}"
            return DRTLooseOutput(False, f"Linear alignment failed: {la.message}", [], [], [], bg_res.bias_g.cpu(), torch.zeros(3), torch.zeros(3), imu_pre_list, diagnostics)

        gr = refine_gravity_on_sphere(
            rotation_body=R_stack,
            position_body=p_stack,
            delta_v=d_v,
            delta_p=d_p,
            dt=dt,
            init_velocity=la.velocity,
            init_gravity=la.gravity,
            g_mag=float(getattr(self.cfg, "g_mag", 9.81)),
            iterations=int(getattr(self.cfg, "gravity_refine_iters", 5)),
        )
        if not gr.success:
            diagnostics["reject_reason"] = f"Gravity refine failed: {gr.message}"
            return DRTLooseOutput(False, f"Gravity refine failed: {gr.message}", [], [], [], bg_res.bias_g.cpu(), torch.zeros(3), la.gravity.cpu(), imu_pre_list, diagnostics)

        max_bg = float(getattr(getattr(self.cfg, "accept_criteria", SimpleNamespace()), "max_bg_norm", 0.5))
        if torch.linalg.vector_norm(bg_res.bias_g).item() > max_bg:
            diagnostics["reject_reason"] = "Gyro bias norm too large"
            return DRTLooseOutput(False, "Gyro bias norm too large", [], [], [], bg_res.bias_g.cpu(), torch.zeros(3), gr.gravity.cpu(), imu_pre_list, diagnostics)

        diagnostics.update(
            {
                "accept": 1.0,
                "bg_norm": float(torch.linalg.vector_norm(bg_res.bias_g).item()),
                "gW_mag": float(torch.linalg.vector_norm(gr.gravity).item()),
                "scale_s": 1.0,
            }
        )
        if logger is not None and log_step is not None:
            logger.log_scalars(
                {
                    "init.drt.window_len": float(K),
                    "init.drt.n_views.total": float(diagnostics["n_views_total"]),
                    "init.drt.parallax_px.p50": float(diagnostics["parallax_p50"]),
                    "init.drt.parallax_px.p5": float(diagnostics["parallax_p5"]),
                    "init.drt.accept": 1.0,
                    "init.drt.bg_solver.residual": float(bg_res.final_loss),
                    "init.drt.bg.norm": float(diagnostics["bg_norm"]),
                    "init.drt.gW.mag": float(diagnostics["gW_mag"]),
                    "init.drt.gW.mag_err": float(abs(diagnostics["gW_mag"] - 9.81007)),
                    "init.drt.scale_s": 1.0,
                },
                int(log_step),
            )

        return DRTLooseOutput(
            ok=True,
            reason="ok",
            rotation=[r.float() for r in R_body],
            position=[p.float() for p in p_body],
            velocity=[v.float() for v in gr.velocity],
            bias_g=bg_res.bias_g.float().cpu(),
            bias_a=torch.zeros(3, dtype=torch.float32),
            gravity=gr.gravity.float().cpu(),
            imu_pres=imu_pre_list,
            diagnostics=diagnostics,
        )

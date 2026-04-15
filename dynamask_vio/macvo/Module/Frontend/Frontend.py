"""
What is Frontend?
    - up to now (2024/06) it's just a combination of StereoDepth and Matcher

Why we need Frontend?
    - Sometime the depth estimation and matching are tightly coupled, so we need a way to combine them.
    
      For instance, if depth (using disparity) and matching uses the same network with same weight, instead of
      inference twice in sequential mannor, we can compose a batch with size of 2 and inference once.
    
How to use this?
    - If there's no specific need (e.g. for performance improvement mentioned above), just use the `FrontendCompose`
      to combine an IStereoDepth and an IMatcher. This should work just fine.
    
    - Otherwise implement a new IFrontend and plug it in the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import time
from pathlib import Path
from types import SimpleNamespace
from typing import overload, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import StereoData
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype

from .StereoDepth import IStereoDepth, disparity_to_depth, disparity_to_depth_cov
from .Matching    import IMatcher

# Frontend interface ###
class IFrontend(ABC, ConfigTestableSubclass):
    """
    Jointly estimate dense depth map, dense matching and potentially their covariances given two pairs of stereo images.
    
    `IFrontend(frame_t1: StereoData, frame_t2: StereoData) -> IStereoDepth.Output, IMatcher.Output`

    Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` where

    * [0] - IStereoDepth.Output, the predicted depth (and potentially depth covariance & validity mask)
    * [1] - IMatcher.Output or None, the predicted flow (potentially flow covariance & mask)
    
    If frame_t1 is None, return only `IStereoDepth.Output` and leave [1] as None.

    #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
    """
    
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
    
    @property
    @abstractmethod
    def provide_cov(self) -> tuple[bool, bool]: ...
    
    @abstractmethod
    def estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]:
        """
        Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` of
        -   [0] - IStereoDepth output of stereo frame from time t2
        -   [1] - IMatcher     output of left camera of t1 -> t2.

        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        ...

    @abstractmethod
    def estimate_depth(self, frame: StereoData) -> IStereoDepth.Output:
        """
        Given stereo frames with imageL, imageR with shape of Bx3xHxW, return IStereoDepth `output` of stereo frame
        
        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        ...

    def estimate_triplet(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IStereoDepth.Output, IMatcher.Output]:
        """
        Given two frames with imageL, imageR with shape of Bx3xHxW, return `output` of
        -   [0] - IStereoDepth output of stereo frame from time t1
        -   [1] - IStereoDepth output of stereo frame from time t2
        -   [2] - IMatcher     output of left camera of t1 -> t2.
        
        #### All outputs maybe padded with `nan` if model can't output prediction with same shape as input image.
        """
        # Here is a simple yet less efficient sequential implementation, feel free to override with a more efficient (e.g. batched inference)
        # approach!
        depth_t1 = self.estimate_depth(frame_t1)
        depth_t2, match_t12 = self.estimate_pair(frame_t1, frame_t2)
        return depth_t1, depth_t2, match_t12
    
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor, interpolate: bool=False) -> torch.Tensor: ...
    @overload
    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: None, interpolate: bool=False) -> None: ...

    @staticmethod
    def retrieve_pixels(pixel_uv: torch.Tensor, scalar_map: torch.Tensor | None, interpolate: bool=False) -> torch.Tensor | None:
        """
        Given a pixel_uv (Nx2) tensor, retrieve the pixel values (CxN) from scalar_map (BxCxHxW).
        
        #### Note that the pixel_uv is in (x, y) format, not (row, col) format.
        
        #### Note that only first sample of scalar_map is used. (Batch idx=0)
        """
        if scalar_map is None: return None
        
        if interpolate:
            raise NotImplementedError("Not implemented yet")
        else:
            values = scalar_map[0, ..., pixel_uv[..., 1].long(), pixel_uv[..., 0].long()]
            return values

# End #######################

@dataclass
class CUDAGraphHandler:
    graph: torch.cuda.CUDAGraph
    shape: torch.Size
    static_input: dict[str, torch.Tensor]
    static_ouput: dict[str, torch.Tensor]

# Implementations

class FrontendCompose(IFrontend):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.depth = IStereoDepth.instantiate(self.config.depth.type, self.config.depth.args)
        self.match = IMatcher.instantiate(self.config.match.type, self.config.match.args)

    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return self.depth.provide_cov, self.match.provide_cov
    
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]:
        return (
            self.depth.estimate(frame_t2),
            self.match.estimate(frame_t1, frame_t2)
        )
    
    def estimate_depth(self, frame: StereoData) -> IStereoDepth.Output:
        return self.depth.estimate(frame)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        IMatcher.is_valid_config(config.match)
        IStereoDepth.is_valid_config(config.depth)


class FlowFormerCovFrontend(IFrontend):
    TENSOR_RT_AOT_RESULT_PATH = Path("./cache/FlowFormerCov_TRTCache")
    T_SUPPORT_DTYPE = Literal["fp32", "bf16", "fp16"]
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerCov import build_flowformer
        
        cfg = get_cfg()
        cfg.latentcostformer.decoder_depth = self.config.decoder_depth
        model = build_flowformer(cfg, reflect_torch_dtype(config.enc_dtype), reflect_torch_dtype(config.dec_dtype))
        ckpt  = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)
        
        model.eval()
        model.to(self.config.device)
        model.load_ddp_state_dict(ckpt)
        self.model = model
    
    @property
    def provide_cov(self) -> tuple[bool, bool]:
        return True, True
    
    @staticmethod
    def inference_2_depth(flow_12: torch.Tensor, cov_12: torch.Tensor, frame: StereoData, enforce_positive_disparity: bool) -> IStereoDepth.Output:
        disparity, disparity_cov = flow_12[:, :1].abs(), cov_12[:, :1]
        depth_map = disparity_to_depth(disparity, frame.frame_baseline, frame.fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, frame.frame_baseline, frame.fx)
        
        if enforce_positive_disparity:
            bad_mask = flow_12[:, :1] <= 0
        else:
            bad_mask = None
        
        return IStereoDepth.Output(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov, mask=bad_mask)

    @staticmethod
    def inference_2_match(flow_12: torch.Tensor, cov_12: torch.Tensor) -> IMatcher.Output:
        match_map, match_cov = flow_12, cov_12
        match_mask = None
        return IMatcher.Output.from_partial_cov(flow=match_map, cov=match_cov, mask=match_mask)

    @torch.inference_mode()
    def estimate_depth(self, frame: StereoData) -> IStereoDepth.Output:
        input_A, input_B = frame.imageL, frame.imageR
        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)

        est_flow, est_cov = self.model.inference(input_A, input_B)
        
        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()
        
        return self.inference_2_depth(est_flow, est_cov, frame, self.config.enforce_positive_disparity)
    
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]:
        input_A = torch.cat([frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t2.imageR, frame_t2.imageL], dim=0)
        
        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)
        
        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()
        
        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )
    
    @torch.inference_mode()
    def estimate_triplet(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IStereoDepth.Output, IMatcher.Output]:
        input_A = torch.cat([frame_t1.imageL, frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t1.imageL, frame_t2.imageR, frame_t2.imageL], dim=0)

        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)
        
        est_flow: torch.Tensor = est_flow.float()
        est_cov : torch.Tensor = est_cov.float()

        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t1, self.config.enforce_positive_disparity),
            self.inference_2_depth(est_flow[1:2], est_cov[1:2], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[2:3], est_cov[2:3])
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight"    : lambda s: isinstance(s, str), # Model Checkpoint path
            "device"    : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "dec_dtype" : lambda b: isinstance(b, str) and b in ("fp32", "fp16", "bf16"),
            "enc_dtype" : lambda b: isinstance(b, str) and b in ("fp32", "fp16", "bf16"),
            "enforce_positive_disparity": lambda b: isinstance(b, bool),
            "decoder_depth" : lambda v: isinstance(v, int)
        })


class CUDAGraph_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowformerCov Frontend, but using CUDAGraph acceleration to improve inference speed.
    """
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        
        self.cuda_graph: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph_FlowFormerCovFrontend can only run on CUDA device."
        
        torch.backends.cuda.matmul.allow_tf32 = True    # Allow tensor cores
        torch.backends.cudnn.allow_tf32 = True          # Allow tensor cores
        torch.set_float32_matmul_precision("medium")    # Reduced precision for higher throughput
        torch.backends.cuda.preferred_linalg_library = "cusolver"   # For faster linalg ops
       
    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]:
        # Joint inference
        input_A = torch.cat([frame_t2.imageL, frame_t1.imageL], dim=0)
        input_B = torch.cat([frame_t2.imageR, frame_t2.imageL], dim=0)
        
        input_A = input_A.to(device=self.config.device)
        input_B = input_B.to(device=self.config.device)

        est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        time.sleep(0.0) # Hint OS scheduler for context switch
        
        est_flow = est_flow.float()
        est_cov  = est_cov.float()
        
        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], frame_t2, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )
    
    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If does not exist a cuda graph
            build one and run inference through it. 
            Store the resulted graph in frontend for future use.
        If does exist a cuda graph
            Launch graph with new input.
        """
        if self.cuda_graph is None:
            Logger.write("info", "Building CUDAGraph for FlowFormerCovFrontend")
            static_input_A, static_input_B   = torch.empty_like(inp_A, device='cuda'), torch.empty_like(inp_A, device='cuda')
            
            static_input_A.copy_(inp_A)
            static_input_B.copy_(inp_B)
            
            output_val: None | torch.Tensor = None
            output_cov: None | torch.Tensor = None
            
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())  #type: ignore
            with torch.cuda.stream(s):                  #type: ignore
                for _ in range(3):
                    output_val, output_cov = self.model.inference(static_input_A, static_input_B)
            torch.cuda.current_stream().wait_stream(s)
            assert output_val is not None and output_cov is not None
            
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output, static_output_cov = self.model.inference(static_input_A, static_input_B)
            
            self.cuda_graph = CUDAGraphHandler(
                graph, inp_A.shape,
                static_input={"input_A": static_input_A, "input_B": static_input_B},
                static_ouput={"flow": static_output, "flow_cov": static_output_cov}
            )
            Logger.write("info", "CUDAGraph Built. Will use CUDAGraph for accelerated inference.")
            
            return output_val, output_cov
        else:
            g_context = self.cuda_graph
            
            assert inp_A.shape == g_context.shape, f"Input shape mismatch for CUDAGraph replay: {inp_A.shape} != {g_context.shape}"
            
            g_context.static_input["input_A"].copy_(inp_A)
            g_context.static_input["input_B"].copy_(inp_B)
            
            g_context.graph.replay()
            time.sleep(0.0) # Hint OS scheduler for context switch
            
            result_val = g_context.static_ouput["flow"].clone()
            result_cov = g_context.static_ouput["flow_cov"].clone()
            
        return result_val, result_cov


class StaticConfidence_FlowFormerCovFrontend(FlowFormerCovFrontend):
    """
    FlowFormerCov frontend with trainable static-confidence head and AirIMU branch.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        from ..Network.AirIMU.encoder import AirIMUEncoder
        from ..Network.DynamicHead.head import StaticConfidenceHead

        self.imu_encoder = AirIMUEncoder.from_config(config.imu)
        self.dynamic_head = StaticConfidenceHead.from_config(config.dynamic_head)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        if getattr(config.dynamic_head, "weight", None):
            ckpt = torch.load(config.dynamic_head.weight, map_location=config.device, weights_only=False)
            head_state = ckpt.get("head_state_dict", ckpt.get("dynamic_head", ckpt.get("state_dict", ckpt)))
            self.dynamic_head.load_state_dict(head_state, strict=False)
            if "feature_mlp" in ckpt:
                self.imu_encoder.feature_mlp.load_state_dict(ckpt["feature_mlp"], strict=False)
            if "temperature" in ckpt:
                self.dynamic_head.T_calib.fill_(float(ckpt["temperature"]))

        self.imu_encoder.to(config.device)
        self.dynamic_head.to(config.device)
        self.imu_encoder.corrector.eval()

        self._h_prev: torch.Tensor | None = None
        self._last_frame_ns: int | None = None
        self._dt_reset_ns = int(getattr(config.dynamic_head, "dt_reset_ms", 500)) * 1_000_000
        self._tau0 = float(getattr(config.dynamic_head, "tau0", 1.0))
        self._alpha = float(getattr(config.dynamic_head, "alpha", 1.0))

    def reset_stream(self) -> None:
        self._h_prev = None
        self._last_frame_ns = None
        self.dynamic_head.reset_state()

    def _maybe_reset(self, frame_ns: int) -> None:
        if self._last_frame_ns is None:
            return
        if frame_ns - self._last_frame_ns > self._dt_reset_ns:
            self.reset_stream()

    @staticmethod
    def _pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([x, y], dim=0).unsqueeze(0)

    @staticmethod
    def _rigid_flow(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        b, _, h, w = depth.shape
        grid = StaticConfidence_FlowFormerCovFrontend._pixel_grid(h, w, depth.device, depth.dtype).expand(b, -1, -1, -1)
        u, v = grid[:, 0], grid[:, 1]

        fx = K[:, 0, 0].view(b, 1, 1)
        fy = K[:, 1, 1].view(b, 1, 1)
        cx = K[:, 0, 2].view(b, 1, 1)
        cy = K[:, 1, 2].view(b, 1, 1)

        z = depth[:, 0].clamp(min=eps)
        x = (u - cx) / fx.clamp(min=eps) * z
        y = (v - cy) / fy.clamp(min=eps) * z
        pts = torch.stack([x, y, z], dim=1).view(b, 3, -1)
        pts2 = (R @ pts) + t.view(b, 3, 1)
        x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2].clamp(min=eps)
        u2 = fx.view(b, 1) * (x2 / z2) + cx.view(b, 1)
        v2 = fy.view(b, 1) * (y2 / z2) + cy.view(b, 1)
        uv2 = torch.stack([u2.view(b, h, w), v2.view(b, h, w)], dim=1)
        return uv2 - grid

    @staticmethod
    def _proxy_valid(depth: torch.Tensor, flow_obs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        b, _, h, w = depth.shape
        grid = StaticConfidence_FlowFormerCovFrontend._pixel_grid(h, w, flow_obs.device, flow_obs.dtype).expand(b, -1, -1, -1)
        uv2 = grid + flow_obs
        in_view = (uv2[:, 0] >= 0) & (uv2[:, 0] <= (w - 1)) & (uv2[:, 1] >= 0) & (uv2[:, 1] <= (h - 1))
        return ((depth[:, 0] > eps) & in_view).unsqueeze(1)

    def _tau_depth(self, depth: torch.Tensor, K: torch.Tensor, delta_p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        focal = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
        depth_safe = depth.clamp(min=eps)
        t_norm = delta_p.norm(dim=-1)
        return self._tau0 + self._alpha * (focal.view(-1, 1, 1, 1) / depth_safe) * t_norm.view(-1, 1, 1, 1)

    def _pack_imu_window(self, frame) -> tuple[torch.Tensor, torch.Tensor]:
        imu = frame.imu
        time_ns = imu.time_ns.to(self.config.device).float()
        if time_ns.ndim == 2:
            time_ns = time_ns.unsqueeze(-1)
        t_rel = (time_ns - time_ns[:, :1, :]) / 1e9
        acc = imu.acc.to(self.config.device).float()
        gyro = imu.gyro.to(self.config.device).float()
        imu_window = torch.cat([t_rel, acc, gyro], dim=-1)
        imu_mask = torch.isfinite(imu_window).all(dim=-1)
        return torch.nan_to_num(imu_window, nan=0.0), imu_mask

    @staticmethod
    def _resolve_imu_timestamp_ns(frame) -> int | None:
        if hasattr(frame, "imu") and getattr(frame, "imu") is not None:
            time_ns = frame.imu.time_ns
            if isinstance(time_ns, torch.Tensor) and time_ns.numel() > 0:
                return int(time_ns.reshape(-1)[-1].item())
        return None

    def estimate_pair(self, frame_t1, frame_t2):
        depth_out, match_out = super().estimate_pair(frame_t1, frame_t2)
        if self.model.last_context is None:
            raise RuntimeError("FlowFormer context feature is unavailable (self.model.last_context is None).")

        f_ctx = self.model.last_context.detach()
        if f_ctx.size(0) >= 2:
            f_ctx = f_ctx[1:2]
        if f_ctx.size(1) != 128:
            raise RuntimeError(f"Expected FlowFormer context channels=128, got {f_ctx.size(1)}")

        h8, w8 = f_ctx.shape[-2:]
        h, w = match_out.flow.shape[-2:]
        flow_lo = F.interpolate(match_out.flow, size=(h8, w8), mode="bilinear", align_corners=False) / 8.0
        cov_full = match_out.cov
        if cov_full is None:
            cov_full = torch.zeros((match_out.flow.size(0), 3, h, w), device=match_out.flow.device, dtype=match_out.flow.dtype)
        cov_lo = F.interpolate(cov_full, size=(h8, w8), mode="bilinear", align_corners=False)
        if cov_lo.size(1) == 2:
            cov_lo = torch.cat([cov_lo, torch.zeros_like(cov_lo[:, :1])], dim=1)
        elif cov_lo.size(1) > 3:
            cov_lo = cov_lo[:, :3]

        imu_window: torch.Tensor | None = None
        imu_out: dict[str, torch.Tensor] | None = None
        if hasattr(frame_t2, "imu_window") and hasattr(frame_t2, "imu_mask"):
            imu_window = frame_t2.imu_window.to(self.config.device)
            imu_mask = frame_t2.imu_mask.to(self.config.device)
            imu_out = self.imu_encoder(imu_window, imu_mask, share_with_backend=True)
            f_imu = imu_out["f_imu"]
            delta_R = imu_out["delta_R"]
            delta_p = imu_out["delta_p"]
        elif hasattr(frame_t2, "imu") and frame_t2.imu is not None:
            imu_window, imu_mask = self._pack_imu_window(frame_t2)
            imu_out = self.imu_encoder(imu_window, imu_mask, share_with_backend=True)
            f_imu = imu_out["f_imu"]
            delta_R = imu_out["delta_R"]
            delta_p = imu_out["delta_p"]
        elif str(getattr(self.config.dynamic_head, "imu_fusion", "film")) == "none":
            b = match_out.flow.size(0)
            f_imu = None
            delta_R = torch.eye(3, device=self.config.device).unsqueeze(0).expand(b, -1, -1).contiguous()
            delta_p = torch.zeros((b, 3), device=self.config.device)
            imu_out = {
                "delta_R": delta_R,
                "delta_v": torch.zeros((b, 3), device=self.config.device),
                "delta_p": delta_p,
                "Sigma_preint": torch.eye(9, device=self.config.device).unsqueeze(0).expand(b, -1, -1).contiguous(),
                "J_R_bg": torch.zeros((b, 3, 3), device=self.config.device),
                "J_v_bg": torch.zeros((b, 3, 3), device=self.config.device),
                "J_v_ba": torch.zeros((b, 3, 3), device=self.config.device),
                "J_p_bg": torch.zeros((b, 3, 3), device=self.config.device),
                "J_p_ba": torch.zeros((b, 3, 3), device=self.config.device),
            }
        else:
            raise RuntimeError("StaticConfidence frontend requires IMU data unless dynamic_head.imu_fusion='none'.")

        depth_full = depth_out.depth.to(self.config.device)
        if hasattr(frame_t2, "K"):
            K_full = frame_t2.K.to(self.config.device)
        else:
            K_full = frame_t2.stereo.K.to(self.config.device)
        flow_full = match_out.flow.to(self.config.device)

        f_rigid_imu = self._rigid_flow(depth_full, K_full, delta_R, delta_p)
        delta_f = flow_full - f_rigid_imu
        r_imu = torch.linalg.vector_norm(delta_f, dim=1, keepdim=True)
        tau_d = self._tau_depth(depth_full, K_full, delta_p).clamp(min=1e-6)
        r_imu_norm = r_imu / tau_d
        valid = self._proxy_valid(depth_full, flow_full).to(dtype=flow_full.dtype)

        proxy_full = torch.cat([delta_f, r_imu, r_imu_norm, valid], dim=1)
        proxy_lo = F.interpolate(proxy_full, size=(h8, w8), mode="bilinear", align_corners=False)
        proxy_lo[:, :2] = proxy_lo[:, :2] / 8.0

        frame_ns = self._resolve_imu_timestamp_ns(frame_t2)
        if frame_ns is None:
            frame_ns = int(frame_t2.frame_ns if hasattr(frame_t2, "frame_ns") else frame_t2.stereo.frame_ns)
        self._maybe_reset(frame_ns)
        logit_lo, h_new = self.dynamic_head(
            f_ctx=f_ctx,
            flow=flow_lo,
            cov=cov_lo,
            f_imu=f_imu,
            proxy=proxy_lo,
            h_prev=self._h_prev,
            depth=F.interpolate(depth_full, size=(h8, w8), mode="nearest"),
            image=F.interpolate(frame_t2.imageL.to(self.config.device), size=(h8 * 2, w8 * 2), mode="bilinear", align_corners=False),
        )
        self._h_prev = h_new.detach()
        self._last_frame_ns = frame_ns

        p_static, p_visible, c_eff = self.dynamic_head.effective_confidence(logit_lo)
        w_lo = self.dynamic_head.map_weight(c_eff)
        c_full = F.interpolate(c_eff, size=(h, w), mode="bilinear", align_corners=False)
        w_full = F.interpolate(w_lo, size=(h, w), mode="bilinear", align_corners=False)
        logit_full = F.interpolate(logit_lo, size=(h, w), mode="bilinear", align_corners=False)
        p_static_full = F.interpolate(p_static, size=(h, w), mode="bilinear", align_corners=False)
        p_visible_full = F.interpolate(p_visible, size=(h, w), mode="bilinear", align_corners=False)

        setattr(match_out, "static_conf", c_full.squeeze(1))
        setattr(match_out, "static_weight", w_full.squeeze(1))
        setattr(match_out, "static_logit", logit_full[:, :1].squeeze(1))
        setattr(match_out, "p_static", p_static_full.squeeze(1))
        setattr(match_out, "p_visible", p_visible_full.squeeze(1))
        assert imu_out is not None
        dt_sum = torch.zeros((match_out.flow.size(0), 1), device=self.config.device)
        if imu_window is not None and imu_window.size(1) > 1:
            dt_sum = (imu_window[:, 1:, 0] - imu_window[:, :-1, 0]).clamp(min=0.0).sum(dim=1, keepdim=True)
        imu_factor = {
            "delta_R": imu_out["delta_R"].detach().cpu(),
            "delta_v": imu_out["delta_v"].detach().cpu(),
            "delta_p": imu_out["delta_p"].detach().cpu(),
            "Sigma_preint": imu_out["Sigma_preint"].detach().cpu(),
            "J_R_bg": imu_out["J_R_bg"].detach().cpu(),
            "J_v_bg": imu_out["J_v_bg"].detach().cpu(),
            "J_v_ba": imu_out["J_v_ba"].detach().cpu(),
            "J_p_bg": imu_out["J_p_bg"].detach().cpu(),
            "J_p_ba": imu_out["J_p_ba"].detach().cpu(),
            "dt": dt_sum.detach().cpu(),
        }
        setattr(match_out, "imu_preintegration", imu_factor)

        return depth_out, match_out

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        super().is_valid_config(config)
        cls._enforce_config_spec(config, {
            "imu": lambda v: v is not None,
            "dynamic_head": lambda v: v is not None,
        }, allow_excessive_cfg=True)

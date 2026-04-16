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
import time
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from typing import overload, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass

from DataLoader import StereoData
from Utility.Device import canonicalize_torch_device, is_supported_device_string
from Utility.PrettyPrint import Logger
from Utility.Timer import Timer
from Utility.Extensions import ConfigTestableSubclass
from Utility.Utils import reflect_torch_dtype

from .StereoDepth import IStereoDepth, disparity_to_depth, disparity_to_depth_cov
from .Matching    import IMatcher


def _cfg_get(config: SimpleNamespace | dict | None, key: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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
        self.config.device = canonicalize_torch_device(self.config.device)
        
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
            "device"    : is_supported_device_string,
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
    FlowFormerCov frontend with an additional static-confidence head.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        from ..Network.DynamicHead import build_head

        self.dynamic_head = build_head(getattr(config, "dynamic_head", None))
        self.dynamic_head = self.dynamic_head.to(config.device)
        self.dynamic_head.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        head_weight = _cfg_get(getattr(config, "dynamic_head", None), "weight", None)
        if isinstance(head_weight, str) and len(head_weight) > 0 and Path(head_weight).exists():
            state = torch.load(head_weight, map_location=config.device, weights_only=True)
            if isinstance(state, dict) and "head_state_dict" in state:
                state = state["head_state_dict"]
            if isinstance(state, dict):
                self.dynamic_head.load_state_dict(state, strict=False)

        self.imu_encoder = None
        imu_cfg = getattr(config, "imu", None)
        if imu_cfg is not None:
            from ..Network.AirIMU.encoder import AirIMUEncoder

            self.imu_encoder = AirIMUEncoder.from_config(imu_cfg).to(config.device)
            self.imu_encoder.eval()
            for param in self.imu_encoder.parameters():
                param.requires_grad_(False)

        self._last_frame_ns: int | None = None
        dt_reset_ms = float(_cfg_get(getattr(config, "dynamic_head", None), "dt_reset_ms", 500.0))
        self._dt_reset_ns = int(max(dt_reset_ms, 0.0) * 1_000_000)
        self._temporal_context_index = int(_cfg_get(getattr(config, "dynamic_head", None), "temporal_context_index", 1))

    def reset_stream(self) -> None:
        self.dynamic_head.reset_state()
        self._last_frame_ns = None

    def _maybe_reset_stream(self, frame_ns: int | None) -> None:
        if frame_ns is None or self._last_frame_ns is None:
            return
        if frame_ns < self._last_frame_ns or (frame_ns - self._last_frame_ns) > self._dt_reset_ns:
            self.reset_stream()

    def _resolve_temporal_context(self, batch_size: int) -> torch.Tensor:
        context = getattr(self.model, "last_context", None)
        if context is None:
            raise RuntimeError("FlowFormerCov model did not expose `last_context`.")

        if context.ndim != 4:
            raise RuntimeError(f"Expected `last_context` with shape [B,C,H,W], got {tuple(context.shape)}")

        if context.shape[0] >= 2 * batch_size:
            return context[batch_size: batch_size * 2].detach().float()

        temporal_idx = min(max(0, self._temporal_context_index), context.shape[0] - 1)
        return context[temporal_idx: temporal_idx + 1].detach().float()

    @staticmethod
    def _prepare_flow_and_cov(flow: torch.Tensor, cov: torch.Tensor, target_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        src_h, src_w = flow.shape[-2:]
        dst_h, dst_w = target_hw
        flow_lo = F.interpolate(flow, size=target_hw, mode="bilinear", align_corners=False)
        cov_lo = F.interpolate(cov, size=target_hw, mode="bilinear", align_corners=False)

        scale_x = float(dst_w) / float(src_w)
        scale_y = float(dst_h) / float(src_h)
        flow_lo[:, 0] *= scale_x
        flow_lo[:, 1] *= scale_y
        return flow_lo, cov_lo

    @staticmethod
    def _rigid_flow_from_delta_pose(
        depth: torch.Tensor,
        K: torch.Tensor,
        delta_R: torch.Tensor,
        delta_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NED camera convention: X is forward depth, Y/Z lateral axes.
        b, _, h, w = depth.shape
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(b, -1, -1)
        elif K.shape[0] == 1 and b > 1:
            K = K.expand(b, -1, -1)

        fx = K[:, 0, 0].view(b, 1, 1)
        fy = K[:, 1, 1].view(b, 1, 1)
        cx = K[:, 0, 2].view(b, 1, 1)
        cy = K[:, 1, 2].view(b, 1, 1)

        v, u = torch.meshgrid(
            torch.arange(h, dtype=depth.dtype, device=depth.device),
            torch.arange(w, dtype=depth.dtype, device=depth.device),
            indexing="ij",
        )
        u = u.view(1, h, w).expand(b, -1, -1)
        v = v.view(1, h, w).expand(b, -1, -1)

        x = depth.squeeze(1).clamp_min(1e-6)
        y = (u - cx) / fx * x
        z = (v - cy) / fy * x
        pts = torch.stack([x, y, z], dim=-1).view(b, -1, 3)

        pts_t = torch.bmm(pts, delta_R.transpose(1, 2)) + delta_p.unsqueeze(1)
        x_t = pts_t[..., 0].clamp_min(1e-6)
        u_t = K[:, 0, 0].view(b, 1) * (pts_t[..., 1] / x_t) + K[:, 0, 2].view(b, 1)
        v_t = K[:, 1, 1].view(b, 1) * (pts_t[..., 2] / x_t) + K[:, 1, 2].view(b, 1)

        flow_x = (u_t - u.reshape(b, -1)).view(b, 1, h, w)
        flow_y = (v_t - v.reshape(b, -1)).view(b, 1, h, w)
        flow = torch.cat([flow_x, flow_y], dim=1)

        valid = (
            torch.isfinite(pts_t).all(dim=-1)
            & (u_t >= 0.0)
            & (u_t <= (w - 1))
            & (v_t >= 0.0)
            & (v_t <= (h - 1))
            & torch.isfinite(x_t)
            & (depth.squeeze(1) > 1e-6)
        )
        return flow, valid.view(b, 1, h, w).to(dtype=depth.dtype)

    @staticmethod
    def _build_proxy_channels(
        depth: torch.Tensor,
        flow_obs: torch.Tensor,
        K: torch.Tensor,
        delta_R: torch.Tensor,
        delta_p: torch.Tensor,
    ) -> torch.Tensor:
        rigid_flow, valid = StaticConfidence_FlowFormerCovFrontend._rigid_flow_from_delta_pose(
            depth=depth,
            K=K,
            delta_R=delta_R,
            delta_p=delta_p,
        )
        delta_f = flow_obs - rigid_flow
        r_imu = torch.linalg.vector_norm(delta_f, dim=1, keepdim=True)

        b = depth.shape[0]
        trans_mag = torch.linalg.vector_norm(delta_p, dim=1).view(b, 1, 1, 1)
        if K.ndim == 3:
            fx = K[:, 0, 0].view(b, 1, 1, 1)
        else:
            fx = K[0, 0].reshape(1, 1, 1, 1).expand(b, -1, -1, -1)
        tau = 1.0 + (fx / depth.clamp_min(1e-3)) * trans_mag
        r_imu_norm = r_imu / tau.clamp_min(1e-3)
        return torch.cat([delta_f, r_imu_norm, valid], dim=1)

    def _extract_imu_feature(
        self,
        frame: StereoData,
        ref_feat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
        if self.dynamic_head.imu_fusion not in {"depth_bin_film", "film", "concat", "cross_attention"}:
            return None, None

        if self.imu_encoder is None:
            return torch.zeros(
                (ref_feat.shape[0], self.dynamic_head.imu_feature_dim),
                dtype=ref_feat.dtype,
                device=ref_feat.device,
            ), None

        imu_window = getattr(frame, "imu_window", None)
        imu_mask = getattr(frame, "imu_mask", None)
        imu_dt = getattr(frame, "imu_dt", None)

        if imu_window is None and hasattr(frame, "imu"):
            imu_data = getattr(frame, "imu")
            if imu_data is not None and hasattr(imu_data, "acc") and hasattr(imu_data, "gyro"):
                imu_window = {"acc": imu_data.acc, "gyro": imu_data.gyro}
                if hasattr(imu_data, "time_ns"):
                    imu_window["time_ns"] = imu_data.time_ns

        if imu_window is None:
            return torch.zeros(
                (ref_feat.shape[0], self.dynamic_head.imu_feature_dim),
                dtype=ref_feat.dtype,
                device=ref_feat.device,
            ), None

        def _to_device(val):
            if isinstance(val, torch.Tensor):
                return val.to(self.config.device)
            if isinstance(val, dict):
                return {k: _to_device(v) for k, v in val.items()}
            return val

        imu_window = _to_device(imu_window)
        imu_mask = _to_device(imu_mask)
        imu_dt = _to_device(imu_dt)

        with torch.no_grad():
            imu_out = self.imu_encoder(imu_window, imu_mask=imu_mask, dt=imu_dt)
        return imu_out["f_imu"].to(device=ref_feat.device, dtype=ref_feat.dtype), imu_out

    @Timer.cpu_timeit("Frontend.estimate")
    @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate_pair(self, frame_t1: StereoData, frame_t2: StereoData) -> tuple[IStereoDepth.Output, IMatcher.Output]:
        depth_out, match_out = super().estimate_pair(frame_t1, frame_t2)
        if match_out.cov is None:
            raise RuntimeError("StaticConfidence_FlowFormerCovFrontend requires matcher covariance from FlowFormerCov.")

        f_ctx = self._resolve_temporal_context(frame_t1.imageL.shape[0]).to(match_out.flow)
        flow_lo, cov_lo = self._prepare_flow_and_cov(match_out.flow, match_out.cov, f_ctx.shape[-2:])
        f_imu, imu_out = self._extract_imu_feature(frame_t2, f_ctx)
        if imu_out is not None and ("delta_R" in imu_out) and ("delta_p" in imu_out):
            proxy_full = self._build_proxy_channels(
                depth=depth_out.depth.to(match_out.flow),
                flow_obs=match_out.flow,
                K=frame_t2.K.to(match_out.flow),
                delta_R=imu_out["delta_R"].to(match_out.flow),
                delta_p=imu_out["delta_p"].to(match_out.flow),
            )
        else:
            proxy_full = torch.zeros(
                (match_out.flow.shape[0], 4, *match_out.flow.shape[-2:]),
                dtype=match_out.flow.dtype,
                device=match_out.flow.device,
            )
        proxy_lo = F.interpolate(proxy_full, size=f_ctx.shape[-2:], mode="bilinear", align_corners=False)
        scale_x = float(f_ctx.shape[-1]) / float(match_out.flow.shape[-1])
        scale_y = float(f_ctx.shape[-2]) / float(match_out.flow.shape[-2])
        proxy_lo[:, 0] *= scale_x
        proxy_lo[:, 1] *= scale_y

        frame_ns = None
        try:
            frame_ns = frame_t2.frame_ns
        except Exception:
            frame_ns = None
        self._maybe_reset_stream(frame_ns)

        head_out = self.dynamic_head(
            f_ctx=f_ctx,
            flow=flow_lo,
            cov=cov_lo,
            f_imu=f_imu,
            proxy=proxy_lo,
            h_prev=None,
            depth=depth_out.depth.to(match_out.flow),
        )

        h_full, w_full = match_out.flow.shape[-2:]
        p_static = F.interpolate(head_out["p_static"], size=(h_full, w_full), mode="bilinear", align_corners=False).squeeze(1)
        p_visible = F.interpolate(head_out["p_visible"], size=(h_full, w_full), mode="bilinear", align_corners=False).squeeze(1)
        c_src = head_out["static_conf"] if "static_conf" in head_out else head_out["c_eff"]
        w_src = head_out.get("static_weight", head_out.get("w", c_src))
        c_eff = F.interpolate(c_src, size=(h_full, w_full), mode="bilinear", align_corners=False).squeeze(1)
        w_map = F.interpolate(w_src, size=(h_full, w_full), mode="bilinear", align_corners=False).squeeze(1)

        match_out.p_static = p_static
        match_out.p_visible = p_visible
        match_out.static_conf = c_eff
        match_out.static_weight = w_map
        if imu_out is not None:
            match_out.imu_preint = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in imu_out.items()
                if key != "f_imu"
            }

        if frame_ns is not None:
            self._last_frame_ns = int(frame_ns)
        return depth_out, match_out

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        super().is_valid_config(config)
        if getattr(config, "dynamic_head", None) is None:
            raise KeyError("StaticConfidence_FlowFormerCovFrontend config requires `dynamic_head`.")
        if getattr(config, "imu", None) is None:
            raise KeyError("StaticConfidence_FlowFormerCovFrontend config requires `imu`.")

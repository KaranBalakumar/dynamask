import typing as T
import torch
import numpy as np
from typing_extensions import Self

from Utility.Extensions import AutoScalingTensor
from .Graph import Scaling_DenseEdge_Multi, Scaling_SparseEdge_Multi, Scaling_SingleEdge

# Define storage of interest
from .Template   import (
    FrameStore, MatchStore , PointStore,
    FrameNode , MatchObs, PointNode ,
)

class VisualMap:
    def __init__(self) -> None:
        self.init_size: T.Final[int]  = 1024
        self.max_pt_obs: T.Final[int] = 5
        self.max_frame_range: T.Final[int] = 2
        
        self.frames = FrameStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "K"          : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "baseline"   : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.float32),
                "pose"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "T_BS"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "vel_w"      : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "bias_g"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "bias_a"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "need_interp": AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.bool),
                "time_ns"    : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.long)
            }
        )
        
        self.points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )
        
        self.map_points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )

        self.match = MatchStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pixel1_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel1_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "obs1_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "obs2_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "pixel1_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel1_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "static_conf"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1.0),
            }
        )

        self.frame2match  = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2map    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.match2frame1 = Scaling_SingleEdge(self.init_size)
        self.match2frame2 = Scaling_SingleEdge(self.init_size)
        self.match2point  = Scaling_SingleEdge(self.init_size)
        self.point2match  = Scaling_SparseEdge_Multi(self.init_size, self.max_pt_obs)

        self.imu_factor = {
            "from_idx": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            "to_idx": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            "delta_R": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "delta_v": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
            "delta_p": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
            "Sigma_preint": AutoScalingTensor((self.init_size, 9, 9), grow_on=0, dtype=torch.float32),
            "Sigma_imu15": AutoScalingTensor((self.init_size, 15, 15), grow_on=0, dtype=torch.float32),
            "J_R_bg": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "J_v_bg": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "J_v_ba": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "J_p_bg": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "J_p_ba": AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
            "dt": AutoScalingTensor((self.init_size, 1), grow_on=0, dtype=torch.float32),
            "inflation": AutoScalingTensor((self.init_size, 1), grow_on=0, dtype=torch.float32, init_val=1.0),
            "failed": AutoScalingTensor((self.init_size, 1), grow_on=0, dtype=torch.bool),
        }
        
        self.frames.register_edge(self.frame2map)
        self.frames.register_edge(self.frame2match)
        self.points.register_edge(self.point2match)
        self.match.register_edge(self.match2point)
        self.match.register_edge(self.match2frame1)
        self.match.register_edge(self.match2frame2)
        

    def get_frame2match(self, frame: FrameNode) -> MatchObs:
        return self.match[self.frame2match.project(frame.index)]

    def get_match2point(self, match: MatchObs) -> PointNode:
        return self.points[self.match2point.project(match.index)]
    
    def get_point2match(self, point: PointNode) -> MatchObs:
        return self.match[self.point2match.project(point.index)]
    
    def get_match2frame1(self, match: MatchObs) -> FrameNode:
        return self.frames[self.match2frame1.project(match.index)]
    
    def get_match2frame2(self, match: MatchObs) -> FrameNode:
        return self.frames[self.match2frame2.project(match.index)]
    
    def get_frame2map(self, frame: FrameNode) -> PointNode:
        return self.map_points[self.frame2map.project(frame.index)]

    def push_imu_factor(
        self,
        from_idx: torch.Tensor,
        to_idx: torch.Tensor,
        payload: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        from_idx = from_idx.reshape(-1).to(dtype=torch.long, device="cpu")
        to_idx = to_idx.reshape(-1).to(dtype=torch.long, device="cpu")
        n = from_idx.shape[0]
        if to_idx.shape[0] == 1 and n > 1:
            to_idx = to_idx.expand(n)
        if from_idx.shape[0] != to_idx.shape[0]:
            raise ValueError("from_idx and to_idx must have same length")

        required = {
            "delta_R": (n, 3, 3),
            "delta_v": (n, 3),
            "delta_p": (n, 3),
            "Sigma_preint": (n, 9, 9),
            "J_R_bg": (n, 3, 3),
            "J_v_bg": (n, 3, 3),
            "J_v_ba": (n, 3, 3),
            "J_p_bg": (n, 3, 3),
            "J_p_ba": (n, 3, 3),
            "dt": (n, 1),
        }
        optional = {
            "Sigma_imu15": (n, 15, 15),
            "inflation": (n, 1),
            "failed": (n, 1),
        }
        for key, shape in required.items():
            if key not in payload:
                raise KeyError(f"IMU payload missing key `{key}`")
            tensor = payload[key].to(device="cpu", dtype=torch.float32)
            if tensor.shape[0] == 1 and n > 1:
                tensor = tensor.expand(*shape).contiguous()
            if tuple(tensor.shape) != shape:
                raise ValueError(f"IMU payload key `{key}` expects shape {shape}, got {tuple(tensor.shape)}")
            payload[key] = tensor
        default_optional = {
            "Sigma_imu15": None,
            "inflation": torch.ones((n, 1), dtype=torch.float32),
            "failed": torch.zeros((n, 1), dtype=torch.bool),
        }
        for key, shape in optional.items():
            tensor = payload.get(key, default_optional[key])
            if tensor is None:
                sigma = payload["Sigma_preint"]
                sigma15 = torch.zeros((n, 15, 15), dtype=torch.float32)
                sigma15[:, 0:3, 0:3] = sigma[:, 0:3, 0:3]
                sigma15[:, 3:6, 3:6] = sigma[:, 3:6, 3:6]
                sigma15[:, 6:9, 6:9] = sigma[:, 6:9, 6:9]
                dt = payload["dt"].clamp(min=1.0e-3)
                eye = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1)
                sigma15[:, 9:12, 9:12] = eye * (dt.view(-1, 1, 1) * 1.0e-3)
                sigma15[:, 12:15, 12:15] = eye * (dt.view(-1, 1, 1) * 1.0e-3)
                tensor = sigma15
            if key == "failed":
                tensor = tensor.to(device="cpu", dtype=torch.bool)
            else:
                tensor = tensor.to(device="cpu", dtype=torch.float32)
            if tensor.shape[0] == 1 and n > 1:
                tensor = tensor.expand(*shape).contiguous()
            if tuple(tensor.shape) != shape:
                raise ValueError(f"IMU payload key `{key}` expects shape {shape}, got {tuple(tensor.shape)}")
            payload[key] = tensor

        start_idx = self.imu_factor["from_idx"].size(0)
        self.imu_factor["from_idx"].push(from_idx)
        self.imu_factor["to_idx"].push(to_idx)
        for key in required | optional:
            self.imu_factor[key].push(payload[key])
        return torch.arange(start_idx, start_idx + n, dtype=torch.long)

    def get_imu_factor(self, from_idx: torch.Tensor, to_idx: torch.Tensor) -> dict[str, torch.Tensor] | None:
        from_scalar = int(from_idx.item()) if from_idx.numel() == 1 else int(from_idx.reshape(-1)[0].item())
        to_scalar = int(to_idx.item()) if to_idx.numel() == 1 else int(to_idx.reshape(-1)[0].item())
        from_all = self.imu_factor["from_idx"].tensor
        to_all = self.imu_factor["to_idx"].tensor
        if from_all.numel() == 0:
            return None
        mask = (from_all == from_scalar) & (to_all == to_scalar)
        if not torch.any(mask):
            return None
        factor_idx = int(torch.nonzero(mask, as_tuple=False)[-1].item())
        return {
            k: self.imu_factor[k][factor_idx:factor_idx + 1]
            for k in self.imu_factor
            if k not in {"from_idx", "to_idx"}
        }

    def serialize(self) -> dict[str, np.ndarray]:
        return (
            self.frames.serialize("frames/")
          | self.points.serialize("points/")
          | self.match.serialize("match/")
          | self.frame2match.serialize("edge/frame2match")
          | self.point2match.serialize("edge/point2match")
          | self.match2point.serialize("edge/match2point")
          | self.match2frame1.serialize("edge/match2frame1")
          | self.match2frame2.serialize("edge/match2frame2")
          | self.frame2map.serialize("edge/frame2map")
          | {f"imu/{k}": v.tensor.cpu().numpy() for k, v in self.imu_factor.items()}
        )
    
    @classmethod
    def deserialize(cls, value: dict[str, np.ndarray]) -> Self:
        map = cls()
        map.frames = map.frames.deserialize("frames/", value)
        map.match  = map.match.deserialize("match/", value)
        map.points = map.points.deserialize("points/", value)
        
        map.frame2match  = map.frame2match.deserialize("edge/frame2match", value)
        map.point2match  = map.point2match.deserialize("edge/point2match", value)
        map.match2point  = map.match2point .deserialize("edge/match2point", value)
        map.match2frame1 = map.match2frame1.deserialize("edge/match2frame1", value)
        map.match2frame2 = map.match2frame2.deserialize("edge/match2frame2", value)
        map.frame2map    = map.frame2map.deserialize("edge/frame2map", value)
        for key in map.imu_factor:
            np_key = f"imu/{key}"
            if np_key in value:
                map.imu_factor[key] = AutoScalingTensor(None, grow_on=0, init_tensor=torch.tensor(value[np_key]))
        return map

    def __repr__(self) -> str:
        imu_size = self.imu_factor["from_idx"].size(0)
        return f"VisualMap(#frame={len(self.frames)}, #point={len(self.points)}, #map={len(self.map_points)}, #imu={imu_size})"

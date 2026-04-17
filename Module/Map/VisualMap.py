import typing as T
import torch
import numpy as np
from typing_extensions import Self

from Utility.Extensions import AutoScalingTensor
from .Graph import Scaling_DenseEdge_Multi, Scaling_SparseEdge_Multi, Scaling_SingleEdge

# Define storage of interest
from .Template   import (
    FrameStore, MatchStore , PointStore, IMUEdgeStore,
    FrameNode , MatchObs, PointNode, IMUEdgeNode,
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
                "vel"        : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
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
                "c"              : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1.0),
            }
        )

        self.imu_edges = IMUEdgeStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "from_frame": AutoScalingTensor((self.init_size,),       grow_on=0, dtype=torch.long),
                "to_frame"  : AutoScalingTensor((self.init_size,),       grow_on=0, dtype=torch.long),
                "delta_R"   : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
                "delta_v"   : AutoScalingTensor((self.init_size, 3),     grow_on=0, dtype=torch.float64),
                "delta_p"   : AutoScalingTensor((self.init_size, 3),     grow_on=0, dtype=torch.float64),
                "Sigma"     : AutoScalingTensor((self.init_size, 9, 9),  grow_on=0, dtype=torch.float64),
                "dt"        : AutoScalingTensor((self.init_size,),       grow_on=0, dtype=torch.float64),
                "bias_ref"  : AutoScalingTensor((self.init_size, 6),     grow_on=0, dtype=torch.float64),
                "J_R_bg"    : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
                "J_v_bg"    : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
                "J_v_ba"    : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
                "J_p_bg"    : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
                "J_p_ba"    : AutoScalingTensor((self.init_size, 3, 3),  grow_on=0, dtype=torch.float64),
            }
        )

        self.frame2match  = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2map    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.match2frame1 = Scaling_SingleEdge(self.init_size)
        self.match2frame2 = Scaling_SingleEdge(self.init_size)
        self.match2point  = Scaling_SingleEdge(self.init_size)
        self.point2match  = Scaling_SparseEdge_Multi(self.init_size, self.max_pt_obs)
        
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

    def get_imu_edge(self, from_frame: int | torch.Tensor, to_frame: int | torch.Tensor) -> IMUEdgeNode | None:
        from_id = int(from_frame.item()) if isinstance(from_frame, torch.Tensor) else int(from_frame)
        to_id = int(to_frame.item()) if isinstance(to_frame, torch.Tensor) else int(to_frame)
        if len(self.imu_edges) == 0:
            return None
        mask = (self.imu_edges.data["from_frame"] == from_id) & (self.imu_edges.data["to_frame"] == to_id)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        return self.imu_edges[idx[:1]]

    def serialize(self) -> dict[str, np.ndarray]:
        return (
            self.frames.serialize("frames/")
          | self.points.serialize("points/")
          | self.match.serialize("match/")
          | self.imu_edges.serialize("imu_edges/")
          | self.frame2match.serialize("edge/frame2match")
          | self.point2match.serialize("edge/point2match")
          | self.match2point.serialize("edge/match2point")
          | self.match2frame1.serialize("edge/match2frame1")
          | self.match2frame2.serialize("edge/match2frame2")
          | self.frame2map.serialize("edge/frame2map")
        )
    
    @classmethod
    def deserialize(cls, value: dict[str, np.ndarray]) -> Self:
        map = cls()
        # Backward compatible loading: old map dumps may not include newly added
        # fields (vel/bias/c/imu edges). Missing fields are default-initialized.
        if any(k.startswith("frames/") for k in value.keys()):
            if "frames/vel" not in value and "frames/pose" in value:
                N = value["frames/pose"].shape[0]
                value["frames/vel"] = np.zeros((N, 3), dtype=np.float32)
            if "frames/bias_g" not in value and "frames/pose" in value:
                N = value["frames/pose"].shape[0]
                value["frames/bias_g"] = np.zeros((N, 3), dtype=np.float32)
            if "frames/bias_a" not in value and "frames/pose" in value:
                N = value["frames/pose"].shape[0]
                value["frames/bias_a"] = np.zeros((N, 3), dtype=np.float32)
            map.frames = map.frames.deserialize("frames/", value)
        if any(k.startswith("match/") for k in value.keys()):
            if "match/c" not in value and "match/pixel1_uv" in value:
                N = value["match/pixel1_uv"].shape[0]
                value["match/c"] = np.ones((N, 1), dtype=np.float32)
            map.match = map.match.deserialize("match/", value)
        if any(k.startswith("points/") for k in value.keys()):
            map.points = map.points.deserialize("points/", value)
        if any(k.startswith("imu_edges/") for k in value.keys()):
            map.imu_edges = map.imu_edges.deserialize("imu_edges/", value)
        
        map.frame2match  = map.frame2match.deserialize("edge/frame2match", value)
        map.point2match  = map.point2match.deserialize("edge/point2match", value)
        map.match2point  = map.match2point .deserialize("edge/match2point", value)
        map.match2frame1 = map.match2frame1.deserialize("edge/match2frame1", value)
        map.match2frame2 = map.match2frame2.deserialize("edge/match2frame2", value)
        map.frame2map    = map.frame2map.deserialize("edge/frame2map", value)
        return map

    def __repr__(self) -> str:
        return f"VisualMap(#frame={len(self.frames)}, #point={len(self.points)}, #map={len(self.map_points)})"

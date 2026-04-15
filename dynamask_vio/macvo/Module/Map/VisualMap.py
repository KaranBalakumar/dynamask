import typing as T
import torch
import numpy as np
from typing_extensions import Self

from Utility.Extensions import AutoScalingTensor
from .Graph import Scaling_DenseEdge_Multi, Scaling_SparseEdge_Multi, Scaling_SingleEdge

# Define storage of interest
from .Template   import (
    FrameStore, MatchStore, PointStore, IMUFactorStore,
    FrameNode, MatchObs, PointNode, IMUFactorNode,
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
                "vel_w"      : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32, init_val=0.0),
                "bias_g"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32, init_val=0.0),
                "bias_a"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32, init_val=0.0),
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
                "static_conf"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1.0),
                "static_weight"  : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1.0),
                "obs1_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "obs2_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "pixel1_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel1_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32)
            }
        )
        self.imu_factors = IMUFactorStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "delta_R"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "delta_v"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "delta_p"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "Sigma_preint": AutoScalingTensor((self.init_size, 9, 9), grow_on=0, dtype=torch.float32),
                "J_R_bg"      : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "J_v_bg"      : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "J_v_ba"      : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "J_p_bg"      : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "J_p_ba"      : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "dt"          : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "b_g_ref"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32, init_val=0.0),
                "b_a_ref"     : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32, init_val=0.0),
                "gravity_w"   : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "sigma_bg_rw" : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1e-3),
                "sigma_ba_rw" : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32, init_val=1e-3),
            },
        )

        self.frame2match  = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2map    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2imu    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.match2frame1 = Scaling_SingleEdge(self.init_size)
        self.match2frame2 = Scaling_SingleEdge(self.init_size)
        self.match2point  = Scaling_SingleEdge(self.init_size)
        self.point2match  = Scaling_SparseEdge_Multi(self.init_size, self.max_pt_obs)
        self.imu2frame1   = Scaling_SingleEdge(self.init_size)
        self.imu2frame2   = Scaling_SingleEdge(self.init_size)
        
        self.frames.register_edge(self.frame2map)
        self.frames.register_edge(self.frame2match)
        self.frames.register_edge(self.frame2imu)
        self.points.register_edge(self.point2match)
        self.match.register_edge(self.match2point)
        self.match.register_edge(self.match2frame1)
        self.match.register_edge(self.match2frame2)
        self.imu_factors.register_edge(self.imu2frame1)
        self.imu_factors.register_edge(self.imu2frame2)
        

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
    
    def get_frame2imu(self, frame: FrameNode) -> IMUFactorNode:
        return self.imu_factors[self.frame2imu.project(frame.index)]

    def get_imu2frame1(self, imu_factor: IMUFactorNode) -> FrameNode:
        return self.frames[self.imu2frame1.project(imu_factor.index)]
    
    def get_imu2frame2(self, imu_factor: IMUFactorNode) -> FrameNode:
        return self.frames[self.imu2frame2.project(imu_factor.index)]

    def serialize(self) -> dict[str, np.ndarray]:
        return (
            self.frames.serialize("frames/")
          | self.points.serialize("points/")
          | self.match.serialize("match/")
          | self.imu_factors.serialize("imu/")
          | self.frame2match.serialize("edge/frame2match")
          | self.point2match.serialize("edge/point2match")
          | self.match2point.serialize("edge/match2point")
          | self.match2frame1.serialize("edge/match2frame1")
          | self.match2frame2.serialize("edge/match2frame2")
          | self.frame2map.serialize("edge/frame2map")
          | self.frame2imu.serialize("edge/frame2imu")
          | self.imu2frame1.serialize("edge/imu2frame1")
          | self.imu2frame2.serialize("edge/imu2frame2")
        )
    
    @classmethod
    def deserialize(cls, value: dict[str, np.ndarray]) -> Self:
        map = cls()
        map.frames = map.frames.deserialize("frames/", value)
        map.match  = map.match.deserialize("match/", value)
        map.points = map.points.deserialize("points/", value)
        map.imu_factors = map.imu_factors.deserialize("imu/", value)
        
        map.frame2match  = map.frame2match.deserialize("edge/frame2match", value)
        map.point2match  = map.point2match.deserialize("edge/point2match", value)
        map.match2point  = map.match2point .deserialize("edge/match2point", value)
        map.match2frame1 = map.match2frame1.deserialize("edge/match2frame1", value)
        map.match2frame2 = map.match2frame2.deserialize("edge/match2frame2", value)
        map.frame2map    = map.frame2map.deserialize("edge/frame2map", value)
        map.frame2imu    = map.frame2imu.deserialize("edge/frame2imu", value)
        map.imu2frame1   = map.imu2frame1.deserialize("edge/imu2frame1", value)
        map.imu2frame2   = map.imu2frame2.deserialize("edge/imu2frame2", value)
        return map

    def __repr__(self) -> str:
        return f"VisualMap(#frame={len(self.frames)}, #point={len(self.points)}, #map={len(self.map_points)}, #imu={len(self.imu_factors)})"

import typing as T
from .Graph import TensorBundle, AutoScalingBundle

# Define storage of interest
FrameFeature = T.Literal[
    "K",            # Nx3x3 , dtype=float32
    "baseline",     # Nx1   , dtype=float32
    "pose",         # Nx7   , dtype=float32, pose of sensor under world frame.
    "T_BS",         # Nx7   , dtype=float32, body-to-sensor SE3 transformation.
    "vel_w",        # Nx3   , dtype=float32, velocity in world frame
    "bias_g",       # Nx3   , dtype=float32, gyro bias
    "bias_a",       # Nx3   , dtype=float32, accel bias
    "need_interp",  # Nx1   , dtype=bool
    "time_ns"       # Nx1   , dtype=long
]
MatchingFeature = T.Literal[
    "pixel1_uv",    # Nx2   , dtype=float32
    "pixel1_d",     # Nx1   , dtype=float32
    "pixel2_uv",    # Nx2   , dtype=float32
    "pixel2_d",     # Nx1   , dtype=float32
    "pixel1_disp",  # Nx1   , dtype=float32
    "pixel2_disp",  # Nx1   , dtype=float32
    "pixel1_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel2_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel1_d_cov" ,# Nx1   , dtype=float32
    "pixel2_d_cov" ,# Nx1   , dtype=float32
    "pixel1_disp_cov",    # Nx1   , dtype=float32
    "pixel2_disp_cov",    # Nx1   , dtype=float32
    "static_conf",    # Nx1   , dtype=float32
    "static_weight",  # Nx1   , dtype=float32
    "obs1_covTc",   # Nx3x3 , dtype=float64
    "obs2_covTc",   # Nx3x3 , dtype=float64
]
PointFeature = T.Literal[
    "pos_Tw",       # Nx3   , dtype=float32
    "cov_Tw",       # Nx3x3 , dtype=float64
    "color" ,       # Nx3   , dtype=uint8
]

IMUFactorFeature = T.Literal[
    "delta_R",      # Nx3x3, dtype=float32
    "delta_v",      # Nx3
    "delta_p",      # Nx3
    "Sigma_preint", # Nx9x9
    "J_R_bg",       # Nx3x3
    "J_v_bg",       # Nx3x3
    "J_v_ba",       # Nx3x3
    "J_p_bg",       # Nx3x3
    "J_p_ba",       # Nx3x3
    "dt",           # Nx1
    "b_g_ref",      # Nx3
    "b_a_ref",      # Nx3
    "gravity_w",    # Nx3
    "sigma_bg_rw",  # Nx1
    "sigma_ba_rw",  # Nx1
]


FrameNode    = TensorBundle[FrameFeature]
FrameStore   = AutoScalingBundle[FrameFeature]

MatchObs     = TensorBundle[MatchingFeature]
MatchStore   = AutoScalingBundle[MatchingFeature]

PointNode    = TensorBundle[PointFeature]
PointStore   = AutoScalingBundle[PointFeature]
IMUFactorNode  = TensorBundle[IMUFactorFeature]
IMUFactorStore = AutoScalingBundle[IMUFactorFeature]

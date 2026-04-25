"""
IMU preintegration kernel for the DRT-loose initializer.

Reference:
  Forster et al., 2017. "On-Manifold Preintegration for Real-Time
  Visual–Inertial Odometry." IEEE Transactions on Robotics, 33(1):1–21.

All arithmetic is done in float64 to keep numerical errors well below
the 1 e-4 thresholds used in the unit tests.

Key conventions
---------------
- gyro / acc supplied to integrate() are **raw** (bias-UNCORRECTED).
- b_g, b_a are stored internally and used to bias-correct every tick.
- ΔR is kept as an SO3 LieTensor (pypose); all other accumulators are
  plain float64 tensors.
- dR_log in PreintResult is a plain (3,) tensor (so3 logarithm), NOT a
  LieTensor, so callers can do arithmetic on it directly.
"""

from __future__ import annotations

import torch
import pypose as pp
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreintResult:
    """Accumulated preintegration state for one IMU segment."""

    dR_log: torch.Tensor    # (3,)   SO3 log of accumulated rotation ΔR
    dV:     torch.Tensor    # (3,)   accumulated Δv in body frame at t_0
    dP:     torch.Tensor    # (3,)   accumulated Δp in body frame at t_0
    sum_dt: float           # total integration time [s]

    # 3×3 bias Jacobians (Forster 2017, Table I)
    J_R_bg: torch.Tensor    # ∂ΔR_log / ∂b_g
    J_V_bg: torch.Tensor    # ∂ΔV     / ∂b_g
    J_V_ba: torch.Tensor    # ∂ΔV     / ∂b_a
    J_P_bg: torch.Tensor    # ∂ΔP     / ∂b_g
    J_P_ba: torch.Tensor    # ∂ΔP     / ∂b_a

    Sigma:  torch.Tensor    # (9,9) covariance [ΔR, ΔV, ΔP]


# ---------------------------------------------------------------------------
# Preintegrator
# ---------------------------------------------------------------------------

class IMUPreintegrator:
    """
    On-manifold IMU preintegrator (Forster 2017).

    Parameters
    ----------
    b_g : (3,) tensor  – initial gyroscope bias  [rad/s]
    b_a : (3,) tensor  – initial accelerometer bias [m/s²]
    gyro_noise : float – continuous gyro noise density  [rad/s/√Hz]
    acc_noise  : float – continuous accel noise density [m/s²/√Hz]
    device : torch.device | None
    """

    def __init__(
        self,
        b_g: torch.Tensor | None = None,
        b_a: torch.Tensor | None = None,
        device: torch.device | None = None,
        gyro_noise: float = 1e-4,
        acc_noise:  float = 1e-3,
    ):
        self.device = device or torch.device("cpu")
        dtype = torch.float64

        self.b_g = (b_g.to(dtype=dtype, device=self.device)
                    if b_g is not None
                    else torch.zeros(3, dtype=dtype, device=self.device))
        self.b_a = (b_a.to(dtype=dtype, device=self.device)
                    if b_a is not None
                    else torch.zeros(3, dtype=dtype, device=self.device))

        # Build 6×6 continuous-time noise power-spectral-density matrix
        # Q_c = diag(σ_g² I₃, σ_a² I₃)
        self._Q_c = torch.zeros(6, 6, dtype=dtype, device=self.device)
        self._Q_c[:3, :3] = torch.eye(3, dtype=dtype, device=self.device) * gyro_noise ** 2
        self._Q_c[3:, 3:] = torch.eye(3, dtype=dtype, device=self.device) * acc_noise  ** 2

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulators to identity / zero (keeps biases intact)."""
        dtype  = torch.float64
        dev    = self.device
        zeros3 = torch.zeros(3, dtype=dtype, device=dev)

        # Accumulated rotation as an SO3 LieTensor (identity = [0,0,0,1])
        self._dR: pp.LieTensor = pp.identity_SO3(dtype=dtype, device=dev)

        # Accumulated velocity and position increments
        self._dV = zeros3.clone()
        self._dP = zeros3.clone()

        # Total elapsed time
        self._sum_dt: float = 0.0

        # Bias Jacobians – all start at zero (Forster 2017, eq. 45–47)
        self._J_R_bg = torch.zeros(3, 3, dtype=dtype, device=dev)
        self._J_V_bg = torch.zeros(3, 3, dtype=dtype, device=dev)
        self._J_V_ba = torch.zeros(3, 3, dtype=dtype, device=dev)
        self._J_P_bg = torch.zeros(3, 3, dtype=dtype, device=dev)
        self._J_P_ba = torch.zeros(3, 3, dtype=dtype, device=dev)

        # 9×9 covariance [ΔR, ΔV, ΔP]
        self._Sigma = torch.zeros(9, 9, dtype=dtype, device=dev)

    def integrate(
        self,
        gyro: torch.Tensor,
        acc:  torch.Tensor,
        dt:   float,
    ) -> None:
        """
        Accumulate one IMU tick.

        Parameters
        ----------
        gyro : (3,) raw gyroscope reading  [rad/s]  (bias-UNCORRECTED)
        acc  : (3,) raw accelerometer reading [m/s²] (bias-UNCORRECTED)
        dt   : time step [s]
        """
        dt    = float(dt)
        dtype = torch.float64
        dev   = self.device

        gyro = gyro.to(dtype=dtype, device=dev)
        acc  = acc.to( dtype=dtype, device=dev)

        # ---- Bias correction -------------------------------------------
        omega_tilde = gyro - self.b_g   # ω̃ = ω - b_g
        a_tilde     = acc  - self.b_a   # ã = a - b_a

        # ---- Current rotation matrix (before this step) ----------------
        R_k = self._dR.matrix()         # (3,3)

        # ---- Rotation increment ----------------------------------------
        # dR_delta = Exp(ω̃ · dt)  in SO3
        dR_delta = pp.so3((omega_tilde * dt).contiguous()).Exp()

        # ---- State propagation (Forster 2017, eq. 35) ------------------
        #  ΔP_{k+1} = ΔP_k + ΔV_k · dt + ½ R_k · ã · dt²
        self._dP = (self._dP
                    + self._dV * dt
                    + 0.5 * (R_k @ a_tilde) * dt ** 2)

        #  ΔV_{k+1} = ΔV_k + R_k · ã · dt
        self._dV = self._dV + (R_k @ a_tilde) * dt

        #  ΔR_{k+1} = ΔR_k ⊗ dR_delta
        self._dR = self._dR @ dR_delta

        # ---- Bias Jacobians (Forster 2017, eq. 45–47) ------------------
        #
        # Save previous values (needed for ΔP Jacobians and J_R_bg step)
        J_R_bg_prev = self._J_R_bg.clone()
        J_V_bg_prev = self._J_V_bg.clone()
        J_V_ba_prev = self._J_V_ba.clone()

        skew_a = pp.vec2skew(a_tilde)   # [ã]×  (3×3)

        # J_R_bg: right Jacobian of SO3 exp maps the gyro bias error
        # into an error in the rotation log.
        # Forster 2017, eq. 45:
        #   J_R_bg_{k+1} = dR_delta^T @ J_R_bg_k  -  Jr(ω̃·dt) · dt
        # The negative sign comes from d(ω̃)/d(b_g) = -I.
        Jr = pp.so3((omega_tilde * dt).contiguous()).Jr()   # (3,3) right Jacobian
        self._J_R_bg = dR_delta.Inv().matrix() @ J_R_bg_prev - Jr * dt

        # J_V_bg: Forster eq. 46
        #   J_V_bg_{k+1} = J_V_bg_k - R_k · [ã]× · J_R_bg_k · dt
        # Uses J_R_bg_k (before update) and R_k (before rotation update).
        self._J_V_bg = J_V_bg_prev - R_k @ skew_a @ J_R_bg_prev * dt

        # J_V_ba: ∂ΔV/∂b_a  (Forster eq. 46, second term)
        #   J_V_ba_{k+1} = J_V_ba_k - R_k · dt
        self._J_V_ba = J_V_ba_prev - R_k * dt

        # J_P_bg: Forster eq. 47
        #   J_P_bg_{k+1} = J_P_bg_k + J_V_bg_k · dt
        #                  - 0.5 · R_k · [ã]× · J_R_bg_k · dt²
        # Uses J_R_bg_k and J_V_bg_k (both before update).
        self._J_P_bg = (self._J_P_bg
                        + J_V_bg_prev * dt
                        - 0.5 * R_k @ skew_a @ J_R_bg_prev * dt ** 2)

        # J_P_ba: Forster eq. 47
        #   J_P_ba_{k+1} = J_P_ba_k + J_V_ba_k · dt - 0.5 · R_k · dt²
        self._J_P_ba = (self._J_P_ba
                        + J_V_ba_prev * dt
                        - 0.5 * R_k * dt ** 2)

        # ---- Covariance propagation (Forster 2017, eq. 48) -------------
        # State order: [δφ (3), δv (3), δp (3)]
        #
        # Continuous F (3-block structure, skipping cross-terms with b_g/b_a):
        #   F = [[-[ω̃]×,  0,     0  ],
        #        [-R[ã]×,  0,     0  ],
        #        [  0,     I,     0  ]]
        #
        # G (noise input):
        #   G = [[-I,  0 ],
        #        [ 0, -R ],
        #        [ 0,  0 ]]
        #
        # Discretised (first-order, Euler): Φ = I + F·dt,  Q_d = G Q_c G^T dt
        skew_omega = pp.vec2skew(omega_tilde)  # [ω̃]×

        # Build F (9×9)
        F = torch.zeros(9, 9, dtype=dtype, device=dev)
        F[0:3, 0:3] = -skew_omega               # δφ ← δφ
        F[3:6, 0:3] = -R_k @ skew_a             # δv ← δφ
        F[6:9, 3:6] = torch.eye(3, dtype=dtype, device=dev)  # δp ← δv

        Phi = torch.eye(9, dtype=dtype, device=dev) + F * dt

        # Build G (9×6)
        G = torch.zeros(9, 6, dtype=dtype, device=dev)
        G[0:3, 0:3] = -torch.eye(3, dtype=dtype, device=dev)   # φ ← η_g
        G[3:6, 3:6] = -R_k                                      # v ← η_a

        Q_d = G @ self._Q_c @ G.t() * dt

        self._Sigma = Phi @ self._Sigma @ Phi.t() + Q_d
        self._sum_dt += dt

    def result(self) -> PreintResult:
        """Return a snapshot of the current accumulated preintegration state."""
        dR_log = self._dR.Log().tensor()   # (3,) plain tensor

        return PreintResult(
            dR_log = dR_log.clone(),
            dV     = self._dV.clone(),
            dP     = self._dP.clone(),
            sum_dt = self._sum_dt,
            J_R_bg = self._J_R_bg.clone(),
            J_V_bg = self._J_V_bg.clone(),
            J_V_ba = self._J_V_ba.clone(),
            J_P_bg = self._J_P_bg.clone(),
            J_P_ba = self._J_P_ba.clone(),
            Sigma  = self._Sigma.clone(),
        )

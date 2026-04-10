"""
DynaMask V2 — full model wiring.

Architecture:
  1. IMU Encoder → f_imu + preintegration outputs (unchanged from V1)
  2. RAFT Feature Encoder (fnet) — shared weights, called on both frames,
     FiLM-conditioned with f_imu at each residual stage
  3. RAFT Context Encoder (cnet) — called on frame t only, separate weights,
     output split into GRU hidden init + motion context
  4. Flow Decoder (RAFT Update Operator) — iterative flow refinement,
     produces internal flow + exported dynamic mask

Forward returns: mask + IMU quantities (no flow exported).
Flow is internal — drives GRU state for mask decoding and feeds BA at train time.

Total: ~4M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BasicEncoder, load_raft_encoder_weights
from .imu_encoder import IMUEncoder
from .film import FiLM, create_film_layers
from .flow_decoder import FlowDecoder


class DynaMaskVIO(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model", {})
        imu_cfg = cfg.get("imu", {})

        imu_feature_dim = model_cfg.get("imu_feature_dim", 128)
        output_dim = model_cfg.get("encoder_output_dim", 128)
        hidden_dim = model_cfg.get("gru_hidden_dim", 128)
        gru_iters = model_cfg.get("gru_iterations", 3)
        corr_levels = model_cfg.get("corr_levels", 4)
        corr_radius = model_cfg.get("corr_radius", 4)

        # ── IMU Encoder (unchanged from V1) ──
        self.imu_encoder = IMUEncoder(
            hidden=imu_cfg.get("noise_corrector_hidden", 64),
            dilations=imu_cfg.get("noise_corrector_dilations", [1, 2, 4]),
            initial_variance=imu_cfg.get("initial_variance", 1e-4),
            imu_feature_dim=imu_feature_dim,
        )

        # ── FiLM layers for feature encoder ──
        self.film_layers = create_film_layers(
            imu_dim=imu_feature_dim,
            stage_channels=[64, 96, 128],
        )

        # ── RAFT Feature Encoder (fnet) — shared, called on both frames ──
        self.feature_encoder = BasicEncoder(
            output_dim=output_dim,
            norm_fn="instance",
            film_layers=self.film_layers,
        )

        # ── RAFT Context Encoder (cnet) — separate weights, frame t only ──
        self.context_encoder = BasicEncoder(
            output_dim=output_dim,
            norm_fn="instance",
            film_layers=None,  # no FiLM on context encoder
        )

        # Context split projections (following RAFT design)
        self.conv_net = nn.Conv2d(output_dim, hidden_dim, 1)  # GRU hidden init
        self.conv_inp = nn.Conv2d(output_dim, hidden_dim, 1)  # motion context

        # ── Flow Decoder (RAFT Update Operator) ──
        self.flow_decoder = FlowDecoder(
            hidden_dim=hidden_dim,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            gru_iters=gru_iters,
        )

        # Load pretrained RAFT weights if available
        raft_ckpt = model_cfg.get("raft_checkpoint", None)
        if raft_ckpt:
            load_raft_encoder_weights(self.feature_encoder, raft_ckpt,
                                      prefix="module.fnet.")
            load_raft_encoder_weights(self.context_encoder, raft_ckpt,
                                      prefix="module.cnet.")

    def forward(self, img_prev: torch.Tensor, img_curr: torch.Tensor,
                imu_window: torch.Tensor, imu_mask: torch.Tensor) -> dict:
        """
        Args:
            img_prev:   [B, 3, H, W]
            img_curr:   [B, 3, H, W]
            imu_window: [B, N, 7] (timestamp_delta, ax, ay, az, gx, gy, gz)
            imu_mask:   [B, N] True = valid IMU sample

        Returns dict:
            dynamic_mask : [B, 1, H, W]  values in [0, 1]
            flow         : [B, 2, H/8, W/8]  internal flow (for BA at train time)
            flow_predictions : list of [B, 2, H/8, W/8] per GRU iteration
            mask_logits  : [B, 1, H/8, W/8]  pre-sigmoid mask at 1/8 res
            delta_bg     : [B, N, 3]
            delta_ba     : [B, N, 3]
            sigma2_g     : [B, N, 3]
            sigma2_a     : [B, N, 3]
            delta_R      : [B, 3, 3]
            delta_v      : [B, 3]
            delta_p      : [B, 3]
            Sigma_preint : [B, 9, 9]
            f_imu        : [B, 128]
        """
        B, _, H, W = img_curr.shape

        # 1. IMU encoding
        imu_out = self.imu_encoder(imu_window, imu_mask)
        f_imu = imu_out["f_imu"]  # [B, 128]

        # 2. Normalise images to [-0.5, 0.5] range (RAFT convention)
        img_prev_norm = 2.0 * (img_prev / 255.0) - 0.5
        img_curr_norm = 2.0 * (img_curr / 255.0) - 0.5

        # 3. Feature extraction (shared encoder, FiLM-conditioned)
        fmap_prev = self.feature_encoder(img_prev_norm, f_imu)  # [B, 128, H/8, W/8]
        fmap_curr = self.feature_encoder(img_curr_norm, f_imu)  # [B, 128, H/8, W/8]

        # 4. Context extraction (frame t only, no FiLM)
        context = self.context_encoder(img_curr_norm)  # [B, 128, H/8, W/8]

        # Split context into GRU init and motion context
        net_init = torch.tanh(self.conv_net(context))  # [B, hidden, H/8, W/8]
        inp = torch.relu(self.conv_inp(context))       # [B, hidden, H/8, W/8]

        # 5. Flow decoder (iterative refinement)
        flow_predictions, mask_logits = self.flow_decoder(
            fmap_prev, fmap_curr, net_init, inp)

        # Final flow from last iteration
        flow = flow_predictions[-1]  # [B, 2, H/8, W/8]

        # 6. Upsample mask to full resolution
        mask_up = F.interpolate(mask_logits, size=(H, W),
                                mode="bilinear", align_corners=False)
        dynamic_mask = torch.sigmoid(mask_up)  # [B, 1, H, W]

        return {
            "dynamic_mask": dynamic_mask,
            "flow": flow,
            "flow_predictions": flow_predictions,
            "mask_logits": mask_logits,
            **imu_out,
        }

    def get_parameter_groups(self, base_lr: float,
                              encoder_lr_multiplier: float = 0.5,
                              imu_lr_multiplier: float = 0.1) -> list:
        """Build parameter groups with differential learning rates.

        - Feature encoder + context encoder: 0.5x (preserve pretrained RAFT features)
        - IMU encoder: 0.1x (preserve Phase 1 pretraining)
        - Flow decoder + mask head + FiLM: 1.0x
        """
        encoder_params = (list(self.feature_encoder.parameters()) +
                          list(self.context_encoder.parameters()) +
                          list(self.conv_net.parameters()) +
                          list(self.conv_inp.parameters()))
        encoder_ids = {id(p) for p in encoder_params}

        imu_params = list(self.imu_encoder.parameters())
        imu_ids = {id(p) for p in imu_params}

        other_params = [p for p in self.parameters()
                        if id(p) not in encoder_ids and id(p) not in imu_ids]

        return [
            {"params": other_params, "lr": base_lr,
             "name": "decoder_film"},
            {"params": encoder_params, "lr": base_lr * encoder_lr_multiplier,
             "name": "encoders"},
            {"params": imu_params, "lr": base_lr * imu_lr_multiplier,
             "name": "imu"},
        ]

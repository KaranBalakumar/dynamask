"""DynaMask V2.5 model wiring."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BasicEncoder, load_raft_encoder_weights
from .film import create_film_layers
from .flow_decoder import FlowDecoder
from .imu_encoder import IMUEncoder


class DynaMaskVIO(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        imu_cfg = cfg.get("imu", {})
        score_cfg = cfg.get("score_head", {})

        imu_feature_dim = int(model_cfg.get("imu_feature_dim", 128))
        output_dim = int(model_cfg.get("encoder_output_dim", 128))
        hidden_dim = int(model_cfg.get("gru_hidden_dim", 128))
        gru_iters = int(model_cfg.get("gru_iterations", 3))
        corr_levels = int(model_cfg.get("corr_levels", 4))
        corr_radius = int(model_cfg.get("corr_radius", 4))
        use_checkpoint = bool(model_cfg.get("encoder_gradient_checkpointing", False))

        self.imu_encoder = IMUEncoder(
            imu_feature_dim=imu_feature_dim,
            airimu_interval=int(imu_cfg.get("airimu_interval", 9)),
            airimu_weights_path=model_cfg.get("airimu_weights_path", None),
            freeze_airimu=bool(model_cfg.get("freeze_airimu", True)),
        )

        self.film_layers = create_film_layers(
            imu_dim=imu_feature_dim,
            stage_channels=[64, 96, 128],
        )

        self.feature_encoder = BasicEncoder(
            output_dim=output_dim,
            norm_fn="instance",
            film_layers=self.film_layers,
            use_checkpoint=use_checkpoint,
        )
        self.context_encoder = BasicEncoder(
            output_dim=output_dim,
            norm_fn="instance",
            film_layers=None,
            use_checkpoint=use_checkpoint,
        )

        self.conv_net = nn.Conv2d(output_dim, hidden_dim, 1)
        self.conv_inp = nn.Conv2d(output_dim, hidden_dim, 1)

        self.flow_decoder = FlowDecoder(
            hidden_dim=hidden_dim,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            gru_iters=gru_iters,
            score_logit_clip=float(score_cfg.get("logit_clip", 10.0)),
        )

        t_init = float(score_cfg.get("temperature", 1.0))
        self.register_buffer("temperature_calib", torch.tensor(max(t_init, 1e-3)))

        raft_ckpt = model_cfg.get("raft_checkpoint", None)
        if raft_ckpt:
            load_raft_encoder_weights(self.feature_encoder, raft_ckpt, prefix="module.fnet.")
            load_raft_encoder_weights(self.context_encoder, raft_ckpt, prefix="module.cnet.")

    def set_temperature(self, temperature: float) -> None:
        self.temperature_calib.fill_(max(float(temperature), 1e-3))

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        imu_window: torch.Tensor,
        imu_mask: torch.Tensor,
    ) -> dict:
        _, _, H, W = img_curr.shape

        imu_out = self.imu_encoder(imu_window, imu_mask)
        f_imu = imu_out["f_imu"]

        img_prev_norm = 2.0 * (img_prev / 255.0) - 0.5
        img_curr_norm = 2.0 * (img_curr / 255.0) - 0.5

        fmap_prev = self.feature_encoder(img_prev_norm, f_imu)
        fmap_curr = self.feature_encoder(img_curr_norm, f_imu)
        context = self.context_encoder(img_curr_norm)

        net_init = torch.tanh(self.conv_net(context))
        inp = torch.relu(self.conv_inp(context))

        flow_predictions, score_logits_per_iter = self.flow_decoder(
            fmap_prev, fmap_curr, net_init, inp
        )
        flow = flow_predictions[-1]
        score_logits_lowres = score_logits_per_iter[-1]

        score_logit = F.interpolate(
            score_logits_lowres,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        temperature = self.temperature_calib.clamp(min=1e-3)
        score_cal = torch.sigmoid(score_logit / temperature)

        return {
            "score_logit": score_logit,
            "score_cal": score_cal,
            "dynamic_mask": score_cal,  # backward-compat alias
            "score_logits_per_iter": score_logits_per_iter,
            "score_logits_lowres": score_logits_lowres,
            "mask_logits": score_logits_lowres,  # backward-compat alias
            "flow": flow,
            "flow_predictions": flow_predictions,
            **imu_out,
        }

    def get_parameter_groups(self, base_lr: float, encoder_lr_multiplier: float = 0.1) -> list[dict]:
        """V2.5 optimizer groups with RAFT layerwise LR decay."""
        seen: set[int] = set()

        def collect_unique_params(*modules: nn.Module) -> list[nn.Parameter]:
            params: list[nn.Parameter] = []
            for module in modules:
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    param_id = id(param)
                    if param_id in seen:
                        continue
                    seen.add(param_id)
                    params.append(param)
            return params

        base_modules = [
            self.flow_decoder,
            self.conv_net,
            self.conv_inp,
            self.film_layers,
            self.imu_encoder.feature_mlp,
        ]
        base_params = collect_unique_params(*base_modules)

        # FiLM layers are attached to feature_encoder, but intentionally run at base_lr.
        # Keep parameter groups disjoint to satisfy torch optimizer constraints.
        fnet_params = collect_unique_params(self.feature_encoder)
        cnet_params = collect_unique_params(self.context_encoder)

        groups: list[dict] = []
        if base_params:
            groups.append({"params": base_params, "lr": base_lr, "name": "v25_base"})
        if fnet_params:
            groups.append(
                {
                    "params": fnet_params,
                    "lr": base_lr * encoder_lr_multiplier,
                    "name": "feature_encoder",
                }
            )
        if cnet_params:
            groups.append(
                {
                    "params": cnet_params,
                    "lr": base_lr * encoder_lr_multiplier,
                    "name": "context_encoder",
                }
            )
        return groups

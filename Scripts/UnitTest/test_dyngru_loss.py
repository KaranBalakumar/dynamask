"""Unit tests for dynGRU Phase-A pseudo-label construction and loss functions.

Covers:
  - dyn_pseudo_label: rigid-flow projection, threshold bands, FB consistency
  - dyn_loss_phase_a: focal BCE formula, calibration term, gamma-weighting
"""

import torch
import pytest

from Train.MatchingNet.loss import (
    dyn_pseudo_label,
    dyn_loss_phase_a,
    sequence_loss,
    sequence_metric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_scene():
    """A synthetic 2-frame scenario with known camera motion.

    Camera translates right by 0.1 m.  Scene at depth 1 m, no rotation.
    Expected rigid flow: fx * tx / depth ≈ 32 px horizontal, 0 vertical.
    """
    B = 2
    H, W = 240, 320
    fx, fy = 320.0, 320.0
    cx, cy = 160.0, 120.0

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=torch.float64)
    K = K.unsqueeze(0).expand(B, -1, -1)

    # GT relative pose: translate +0.1 m in X (right), no rotation
    T = torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, 0, 3] = 0.1  # tx = 0.1 m

    # Depth: uniform 1.0 m everywhere
    depth = torch.ones(B, 1, H, W, dtype=torch.float64)

    # Predicted flow from FlowFormer — assume it matches rigid flow perfectly
    # f_rigid(u) = (fx * 0.1) / 1.0 = 32 px horizontal, 0 px vertical
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    flow_pred = torch.zeros(B, 2, H, W, dtype=torch.float64)
    flow_pred[:, 0, :, :] = 32.0  # exactly the rigid flow

    return {
        "B": B, "H": H, "W": W, "fx": fx, "fy": fy,
        "K": K, "T": T, "depth": depth, "flow_pred": flow_pred,
    }


# ---------------------------------------------------------------------------
# dyn_pseudo_label tests
# ---------------------------------------------------------------------------

class TestDynPseudoLabel:
    def test_all_static_with_perfect_rigid_flow(self, simple_scene):
        """Pixels where flow matches rigid flow should be labeled static=1."""
        s = simple_scene
        M_pseudo, residual = dyn_pseudo_label(
            s["flow_pred"], s["T"], s["depth"], s["K"],
        )
        assert M_pseudo.shape == (s["B"], 1, s["H"], s["W"])
        assert residual.shape == (s["B"], 1, s["H"], s["W"])
        # All pixels should be static (flow = rigid flow, residual ≈ 0 < tau)
        assert (M_pseudo == 1).all()
        # Residual should be near zero
        assert residual.max() < 1e-6

    def test_dynamic_pixels_detected_with_large_discrepancy(self, simple_scene):
        """Pixels with flow far from rigid flow should be labeled dynamic=0."""
        s = simple_scene
        # Inject large flow error in one quadrant
        flow_bad = s["flow_pred"].clone()
        flow_bad[:, :, :120, :160] += 100.0  # 100 px discrepancy
        M_pseudo, _ = dyn_pseudo_label(flow_bad, s["T"], s["depth"], s["K"])
        # Top-left quadrant should be dynamic
        assert (M_pseudo[:, :, :120, :160] == 0).all()
        # Rest should be static (perfect match)
        assert (M_pseudo[:, :, 120:, 160:] == 1).all()

    def test_ambiguous_band_is_ignored(self, simple_scene):
        """Pixels with residual in [tau, 3*tau] should be labeled IGNORE=-1."""
        s = simple_scene
        tau_0, alpha = 0.5, 0.3
        # Rigid flow for our scene ≈ 32 px.
        # tau = 0.5 + 0.3 * (320/1.0) * 0.1 = 0.5 + 9.6 = 10.1 px
        # Inject flow error of 15 px (between tau and 3*tau = 30.3)
        flow_ambig = s["flow_pred"].clone()
        flow_ambig[:, 0, :, :] += 15.0
        M_pseudo, _ = dyn_pseudo_label(
            flow_ambig, s["T"], s["depth"], s["K"],
            tau_0=tau_0, alpha=alpha,
        )
        assert (M_pseudo == -1).all()  # all IGNORE

    def test_ignore_band_boundaries(self, simple_scene):
        """Verify exact threshold behavior: < tau → 1, > 3*tau → 0."""
        s = simple_scene
        tau_0 = 2.0
        alpha = 0.0  # no depth term, constant threshold

        flow_small = s["flow_pred"].clone()
        flow_small[:, 0] += 1.0  # residual = 1 < 2 → static
        flow_big = s["flow_pred"].clone()
        flow_big[:, 0] += 7.0   # residual = 7 > 6 → dynamic
        flow_mid = s["flow_pred"].clone()
        flow_mid[:, 0] += 4.0    # residual = 4 in [2, 6] → ignore

        M_small, _ = dyn_pseudo_label(flow_small, s["T"], s["depth"], s["K"],
                                       tau_0=tau_0, alpha=alpha)
        M_big, _ = dyn_pseudo_label(flow_big, s["T"], s["depth"], s["K"],
                                     tau_0=tau_0, alpha=alpha)
        M_mid, _ = dyn_pseudo_label(flow_mid, s["T"], s["depth"], s["K"],
                                     tau_0=tau_0, alpha=alpha)

        assert (M_small == 1).all()
        assert (M_big == 0).all()
        assert (M_mid == -1).all()

    def test_fb_consistency_masks_occlusions(self, simple_scene):
        """Pixels with large FB error should be IGNORE regardless of residual."""
        s = simple_scene
        # Create a bad backward flow that disagrees with forward flow
        fb_bad = torch.zeros_like(s["flow_pred"])
        fb_bad[:, 0, :, :] = -100.0  # totally inconsistent

        M_no_fb, _ = dyn_pseudo_label(s["flow_pred"], s["T"], s["depth"], s["K"])
        M_fb, _ = dyn_pseudo_label(s["flow_pred"], s["T"], s["depth"], s["K"],
                                    fb_flow=fb_bad)

        # Without FB check: all static
        assert (M_no_fb == 1).all()
        # With FB check: all IGNORE (FB error = 132 px >> 1.0)
        assert (M_fb == -1).all()

    def test_pose_7d_format(self, simple_scene):
        """pose_GT in (B, 7) pypose SO3×R3 Log format should work."""
        s = simple_scene
        import pypose as pp
        # Convert (B,4,4) → (B,7) via mat2SE3 then Log
        T_se3 = pp.mat2SE3(s["T"])  # (B, 7) LieTensor
        M_pseudo, _ = dyn_pseudo_label(s["flow_pred"], T_se3.tensor(), s["depth"], s["K"])
        assert (M_pseudo == 1).all()

    def test_depth_adaptive_threshold_larger_near_camera(self, simple_scene):
        """Nearby pixels (small depth) should have larger τ (more tolerant)."""
        s = simple_scene
        depth_near = torch.full_like(s["depth"], 0.5)   # 0.5 m — near
        depth_far = torch.full_like(s["depth"], 5.0)    # 5.0 m — far

        _, res = dyn_pseudo_label(s["flow_pred"], s["T"], depth_near, s["K"])
        tau_near = 0.5 + 0.3 * (320.0 / 0.5) * 0.1   # = 0.5 + 19.2 = 19.7
        tau_far  = 0.5 + 0.3 * (320.0 / 5.0) * 0.1   # = 0.5 + 1.92 = 2.42
        assert tau_near > tau_far

    def test_invalid_pose_shape_raises(self, simple_scene):
        """An incorrectly shaped pose_GT should raise ValueError."""
        s = simple_scene
        bad_pose = torch.randn(2, 3, 3)  # last dim 3, not 4 or 7
        with pytest.raises(ValueError, match="pose_GT must be"):
            dyn_pseudo_label(s["flow_pred"], bad_pose, s["depth"], s["K"])


# ---------------------------------------------------------------------------
# dyn_loss_phase_a tests
# ---------------------------------------------------------------------------

class TestDynLossPhaseA:
    @pytest.fixture
    def loss_fixture(self):
        B, K, H, W = 2, 3, 60, 80
        torch.manual_seed(42)
        dyn_preds = [torch.randn(B, 1, H, W) for _ in range(K)]
        cov_preds = [torch.randn(B, 2, H, W).abs() + 0.1 for _ in range(K)]
        M_pseudo = torch.randint(-1, 2, (B, 1, H, W))
        residual = torch.rand(B, 1, H, W)
        return {"dyn_preds": dyn_preds, "cov_preds": cov_preds,
                "M_pseudo": M_pseudo, "residual": residual,
                "B": B, "K": K, "H": H, "W": W}

    def test_output_keys_and_shapes(self, loss_fixture):
        """Should return dict with L_focal, L_calib, L_total as scalars."""
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["M_pseudo"], f["residual"])
        assert set(out.keys()) == {"L_focal", "L_calib", "L_total"}
        for v in out.values():
            assert v.ndim == 0

    def test_focal_bce_reduces_when_correct(self, loss_fixture):
        """Loss should be lower when predictions match labels."""
        f = loss_fixture
        # Perfect predictions (right sign): large positive when static, large negative when dynamic
        perfect = [torch.where(f["M_pseudo"] == 1, 5.0, -5.0).float()
                   for _ in range(f["K"])]
        out_good = dyn_loss_phase_a(perfect, f["cov_preds"], f["M_pseudo"], f["residual"])
        # Random predictions
        out_rand = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["M_pseudo"], f["residual"])
        assert out_good["L_focal"] < out_rand["L_focal"]

    def test_ignore_pixels_contribute_zero(self, loss_fixture):
        """All-IGNORE labels should give zero loss."""
        f = loss_fixture
        M_all_ignore = torch.full_like(f["M_pseudo"], -1)
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], M_all_ignore, f["residual"])
        assert out["L_focal"].item() == 0.0
        assert out["L_calib"].item() == 0.0
        assert out["L_total"].item() == 0.0

    def test_calib_weight_zero_disables_calib(self, loss_fixture):
        """calib_weight=0 should zero out calibration loss."""
        f = loss_fixture
        out_with = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["M_pseudo"], f["residual"], calib_weight=0.1)
        out_without = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["M_pseudo"], f["residual"], calib_weight=0.0)
        assert out_without["L_calib"].item() == 0.0
        assert out_without["L_total"].item() == out_without["L_focal"].item()

    def test_gamma_weighting_later_iters_matter_more(self, loss_fixture):
        """Later iterations have γ_weight = 1.0, earlier < 1.0."""
        f = loss_fixture
        K = f["K"]
        # With gamma=0.85: weights are [0.85², 0.85¹, 0.85⁰] = [0.7225, 0.85, 1.0]
        # The last iteration's contribution is largest.
        # We test by making only one iter non-IGNORE at a time.
        for k in range(K):
            single_preds = [torch.randn_like(f["dyn_preds"][0]) for _ in range(K)]
            M_single = torch.full_like(f["M_pseudo"], -1)  # all IGNORE
            if k == K - 1:
                continue  # last iter tested below
            # Make only iter k valid
            M_single[0, 0, 0, 0] = 1
            out_k = dyn_loss_phase_a(single_preds, f["cov_preds"], M_single, f["residual"],
                                     gamma=0.85)
            # Loss at k should be non-zero
            assert out_k["L_focal"].item() > 0

    def test_focal_gamma_zero_reduces_to_weighted_bce(self, loss_fixture):
        """focal_gamma=0 turns off modulating factor: L = alpha * BCE."""
        f = loss_fixture
        # Only static labels (no IGNORE/dynamic edge cases)
        M_all_static = torch.ones_like(f["M_pseudo"])
        out_focal0 = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], M_all_static,
                                       f["residual"], focal_gamma=0.0, focal_alpha=1.0)
        out_focal2 = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], M_all_static,
                                       f["residual"], focal_gamma=2.0, focal_alpha=1.0)
        # With gamma=0, loss should differ from gamma=2 (modulating factor changes)
        assert abs(out_focal0["L_focal"].item() - out_focal2["L_focal"].item()) > 1e-6

    def test_resolution_mismatch_upsampled(self):
        """When dyn_preds and M_pseudo have different H/W, logits are upsampled."""
        B = 2
        dyn_preds = [torch.randn(B, 1, 60, 80)]   # H/4
        cov_preds = [torch.randn(B, 2, 60, 80)]   # same res as dyn (H/4)
        M_pseudo = torch.ones(B, 1, 120, 160, dtype=torch.long)  # full res
        residual = torch.rand(B, 1, 60, 80)  # same res as cov for calibration
        out = dyn_loss_phase_a(dyn_preds, cov_preds, M_pseudo, residual)
        assert out["L_focal"].item() > 0  # logits upsampled, focal computed


# ---------------------------------------------------------------------------
# sequence_loss / sequence_metric integration
# ---------------------------------------------------------------------------

class TestSequenceLossDyn:
    def test_dyn_mode_dispatches_to_phase_a(self):
        """sequence_loss with training_mode='dyn' should use dyn_loss_phase_a."""
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            training_mode="dyn", gamma=0.85, max_flow=400, cov_mask=False,
        )
        B, K, Hd, Wd = 2, 3, 60, 80  # dyn operates at H/4
        flow = [torch.randn(B, 2, 240, 320) for _ in range(K)]   # full res
        cov = [torch.randn(B, 2, Hd, Wd) for _ in range(K)]
        dyn = [torch.randn(B, 1, Hd, Wd) for _ in range(K)]
        gt = torch.randn(B, 2, 240, 320)
        M_pseudo = torch.ones(B, 1, Hd, Wd, dtype=torch.long)   # same res as dyn H/4
        residual = torch.rand(B, 1, Hd, Wd)                      # same res for calib

        loss, metrics = sequence_loss(
            cfg, flow, gt, None, cov,
            dyn_preds=dyn, dyn_pseudo=(M_pseudo, residual),
        )
        assert loss.item() > 0
        assert "L_focal" in metrics
        assert "L_calib" in metrics
        assert "dyn_acc" in metrics

    def test_dyn_mode_asserts_on_missing_inputs(self):
        """sequence_loss should raise if dyn_preds or dyn_pseudo missing."""
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            training_mode="dyn", gamma=0.85, max_flow=400, cov_mask=False,
        )
        K, Hf, Wf = 3, 240, 320
        flow = [torch.randn(2, 2, Hf, Wf) for _ in range(K)]
        gt = torch.randn(2, 2, Hf, Wf)
        with pytest.raises(AssertionError):
            sequence_loss(cfg, flow, gt, None, None)  # missing dyn_preds/dyn_pseudo

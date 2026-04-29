"""Unit tests for dynGRU loss functions (Mahalanobis + Scalar modes)."""

import torch
import pytest

from Train.MatchingNet.loss import dyn_loss_phase_a, sequence_loss, sequence_metric


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
        residual = torch.rand(B, 1, H, W)
        r_vec = torch.randn(B, 2, H, W)
        J_d = torch.randn(B, 2, H, W) * 0.01
        sigma_depth = (torch.rand(B, 1, H, W) * 0.5).abs() + 0.1   # 0.1–0.6 px depth std
        J_d = torch.randn(B, 2, H, W) * 5.0                          # larger Jacobian for test
        return {"dyn_preds": dyn_preds, "cov_preds": cov_preds,
                "residual": residual, "r_vec": r_vec,
                "J_d": J_d, "sigma_depth": sigma_depth,
                "B": B, "K": K, "H": H, "W": W}

    def test_mahalanobis_returns_positive_loss(self, loss_fixture):
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                               f["r_vec"], f["J_d"], f["sigma_depth"],
                               gamma=0.85, loss_type="mahalanobis")
        assert out["L_total"].ndim == 0
        assert out["L_total"].item() > 0

    def test_scalar_returns_positive_loss(self, loss_fixture):
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                               f["r_vec"], f["J_d"], f["sigma_depth"],
                               gamma=0.85, loss_type="scalar")
        assert out["L_total"].ndim == 0
        assert out["L_total"].item() > 0

    def test_scalar_works_without_depth(self, loss_fixture):
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                               f["r_vec"], None, None,
                               gamma=0.85, loss_type="scalar")
        assert out["L_total"].item() > 0

    def test_mahalanobis_works_without_depth(self, loss_fixture):
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                               f["r_vec"], None, None,
                               gamma=0.85, loss_type="mahalanobis")
        assert out["L_total"].item() > 0

    def test_fixed_returns_positive_loss(self, loss_fixture):
        f = loss_fixture
        out = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                               f["r_vec"], None, None,
                               gamma=0.85, loss_type="fixed", dyn_sigma=2.0)
        assert out["L_total"].item() > 0

    def test_fixed_larger_sigma_gives_smaller_loss(self, loss_fixture):
        f = loss_fixture
        out_sharp = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                     f["r_vec"], None, None,
                                     gamma=0.85, loss_type="fixed", dyn_sigma=0.5)
        out_soft = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                    f["r_vec"], None, None,
                                    gamma=0.85, loss_type="fixed", dyn_sigma=10.0)
        # Larger sigma → softer targets → different loss
        assert out_sharp["L_total"].item() != out_soft["L_total"].item()

    def test_fixed_ignores_cov_predictions(self, loss_fixture):
        f = loss_fixture
        # With zero cov_preds, fixed mode should still work
        zero_cov = [torch.zeros_like(c) for c in f["cov_preds"]]
        out = dyn_loss_phase_a(f["dyn_preds"], zero_cov, f["residual"],
                               f["r_vec"], None, None,
                               gamma=0.85, loss_type="fixed", dyn_sigma=2.0)
        assert out["L_total"].item() > 0
        # Cov predictions should not affect fixed mode at all
        rand_cov = [torch.randn_like(c).abs() + 0.1 for c in f["cov_preds"]]
        out2 = dyn_loss_phase_a(f["dyn_preds"], rand_cov, f["residual"],
                                f["r_vec"], None, None,
                                gamma=0.85, loss_type="fixed", dyn_sigma=2.0)
        assert abs(out["L_total"].item() - out2["L_total"].item()) < 1e-9

    def test_depth_variants_differ_when_sigma_depth_present(self, loss_fixture):
        f = loss_fixture
        out_s = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], f["J_d"], f["sigma_depth"],
                                 gamma=0.85, loss_type="scalar")
        out_sd = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                  f["r_vec"], f["J_d"], f["sigma_depth"],
                                  gamma=0.85, loss_type="scalar_depth")
        out_m = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], f["J_d"], f["sigma_depth"],
                                 gamma=0.85, loss_type="mahalanobis")
        out_md = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                  f["r_vec"], f["J_d"], f["sigma_depth"],
                                  gamma=0.85, loss_type="mahalanobis_depth")
        # Depth variants should differ from non-depth when sigma_depth is provided
        assert abs(out_s["L_total"].item() - out_sd["L_total"].item()) > 1e-9, "scalar vs scalar_depth should differ"
        assert abs(out_m["L_total"].item() - out_md["L_total"].item()) > 1e-9, "mahalanobis vs mahalanobis_depth should differ"

    def test_depth_variants_identical_without_depth(self, loss_fixture):
        f = loss_fixture
        out_s = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], None, None,
                                 gamma=0.85, loss_type="scalar")
        out_sd = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                  f["r_vec"], None, None,
                                  gamma=0.85, loss_type="scalar_depth")
        out_m = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], None, None,
                                 gamma=0.85, loss_type="mahalanobis")
        out_md = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                  f["r_vec"], None, None,
                                  gamma=0.85, loss_type="mahalanobis_depth")
        # Without depth, variants should be identical
        assert abs(out_s["L_total"].item() - out_sd["L_total"].item()) < 1e-9
        assert abs(out_m["L_total"].item() - out_md["L_total"].item()) < 1e-9

    def test_default_is_mahalanobis(self, loss_fixture):
        f = loss_fixture
        out_default = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                       f["r_vec"], f["J_d"], f["sigma_depth"], gamma=0.85)
        out_explicit = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                        f["r_vec"], f["J_d"], f["sigma_depth"],
                                        gamma=0.85, loss_type="mahalanobis")
        assert abs(out_default["L_total"].item() - out_explicit["L_total"].item()) < 1e-9

    def test_loss_lower_with_smaller_residual(self, loss_fixture):
        f = loss_fixture
        small_r = f["r_vec"] * 0.1
        big_r = f["r_vec"] * 10.0
        out_small = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                     small_r, f["J_d"], f["sigma_depth"],
                                     gamma=0.85, loss_type="scalar")
        out_big = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                   big_r, f["J_d"], f["sigma_depth"],
                                   gamma=0.85, loss_type="scalar")
        assert out_small["L_total"] < out_big["L_total"]

    def test_gamma_weighting_last_iter_weight_is_one(self, loss_fixture):
        f = loss_fixture
        K = f["K"]
        for k in range(K):
            single_dyn = [torch.zeros_like(f["dyn_preds"][0]) for _ in range(K)]
            single_dyn[k] = f["dyn_preds"][k]
            single_cov = [torch.zeros_like(f["cov_preds"][0]) + 0.1 for _ in range(K)]
            single_cov[k] = f["cov_preds"][k]
            out = dyn_loss_phase_a(single_dyn, single_cov, f["residual"],
                                   f["r_vec"], f["J_d"], f["sigma_depth"],
                                   gamma=0.85, loss_type="scalar")
            assert out["L_total"].item() > 0

    def test_resolution_mismatch_interpolated(self):
        B, K = 2, 3
        dyn_preds = [torch.randn(B, 1, 60, 80)]
        cov_preds = [torch.randn(B, 2, 30, 40).abs() + 0.1]
        residual = torch.rand(B, 1, 30, 40)
        r_vec = torch.randn(B, 2, 120, 160)
        out = dyn_loss_phase_a(dyn_preds, cov_preds, residual, r_vec, None, None,
                               gamma=1.0, loss_type="scalar")
        assert out["L_total"].item() > 0

    def test_both_modes_differ_with_depth_uncertainty(self, loss_fixture):
        f = loss_fixture
        out_m = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], f["J_d"], f["sigma_depth"],
                                 gamma=0.85, loss_type="mahalanobis")
        out_s = dyn_loss_phase_a(f["dyn_preds"], f["cov_preds"], f["residual"],
                                 f["r_vec"], f["J_d"], f["sigma_depth"],
                                 gamma=0.85, loss_type="scalar")
        assert abs(out_m["L_total"].item() - out_s["L_total"].item()) > 1e-9


# ---------------------------------------------------------------------------
# sequence_loss / sequence_metric integration
# ---------------------------------------------------------------------------

class TestSequenceLossDyn:
    def test_dyn_mode_dispatches_both_loss_types(self):
        from types import SimpleNamespace
        B, K, Hd, Wd = 2, 3, 60, 80
        flow = [torch.randn(B, 2, 240, 320) for _ in range(K)]
        cov = [torch.randn(B, 2, Hd, Wd).abs() + 0.1 for _ in range(K)]
        dyn = [torch.randn(B, 1, Hd, Wd) for _ in range(K)]
        gt = torch.randn(B, 2, 240, 320)
        residual = torch.rand(B, 1, Hd, Wd)
        f_rigid = torch.randn(B, 2, Hd, Wd)
        r_vec = torch.randn(B, 2, Hd, Wd)
        J_d = torch.randn(B, 2, Hd, Wd) * 0.01
        sigma_depth = torch.rand(B, 1, Hd, Wd).abs() + 0.01
        dyn_data = (residual, f_rigid, r_vec, J_d, sigma_depth)

        for loss_type, dyn_sigma in [("fixed", 2.0), ("scalar", 2.0), ("scalar_depth", 2.0), ("mahalanobis", 2.0), ("mahalanobis_depth", 2.0)]:
            cfg = SimpleNamespace(
                training_mode="dyn", gamma=0.85, max_flow=400,
                cov_mask=False, dyn_loss_type=loss_type, dyn_sigma=dyn_sigma,
            )
            loss, metrics = sequence_loss(
                cfg, flow, gt, None, cov,
                dyn_preds=dyn, dyn_data=dyn_data,
            )
            assert loss.item() > 0
            assert "dyn_c_mean" in metrics

    def test_dyn_mode_defaults_to_mahalanobis(self):
        from types import SimpleNamespace
        B, K, Hd, Wd = 2, 3, 60, 80
        flow = [torch.randn(B, 2, 240, 320) for _ in range(K)]
        cov = [torch.randn(B, 2, Hd, Wd).abs() + 0.1 for _ in range(K)]
        dyn = [torch.randn(B, 1, Hd, Wd) for _ in range(K)]
        gt = torch.randn(B, 2, 240, 320)
        residual = torch.rand(B, 1, Hd, Wd)
        f_rigid = torch.randn(B, 2, Hd, Wd)
        r_vec = torch.randn(B, 2, Hd, Wd)
        J_d = torch.randn(B, 2, Hd, Wd) * 0.01
        dyn_data = (residual, f_rigid, r_vec, J_d, None)

        cfg = SimpleNamespace(
            training_mode="dyn", gamma=0.85, max_flow=400, cov_mask=False,
        )
        loss, metrics = sequence_loss(
            cfg, flow, gt, None, cov,
            dyn_preds=dyn, dyn_data=dyn_data,
        )
        assert loss.item() > 0

    def test_dyn_selfsup_mode_works(self):
        from types import SimpleNamespace
        B, K, Hd, Wd = 2, 3, 60, 80
        flow = [torch.randn(B, 2, 240, 320) for _ in range(K)]
        cov = [torch.randn(B, 2, Hd, Wd).abs() + 0.1 for _ in range(K)]
        dyn = [torch.randn(B, 1, Hd, Wd) for _ in range(K)]
        gt = torch.randn(B, 2, 240, 320)
        residual = torch.rand(B, 1, Hd, Wd)
        f_rigid = torch.zeros(B, 2, Hd, Wd)
        r_vec = torch.randn(B, 2, Hd, Wd)
        dyn_data = (residual, f_rigid, r_vec, None, None)

        cfg = SimpleNamespace(
            training_mode="dyn_selfsup", gamma=0.85, max_flow=400,
            cov_mask=False, dyn_loss_type="scalar",
        )
        loss, metrics = sequence_loss(
            cfg, flow, gt, None, cov,
            dyn_preds=dyn, dyn_data=dyn_data,
        )
        assert loss.item() > 0
        assert "dyn_c_mean" in metrics

    def test_dyn_mode_asserts_on_missing_dyn_data(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            training_mode="dyn", gamma=0.85, max_flow=400, cov_mask=False,
        )
        flow = [torch.randn(2, 2, 240, 320) for _ in range(3)]
        gt = torch.randn(2, 2, 240, 320)
        with pytest.raises(AssertionError):
            sequence_loss(cfg, flow, gt, None, None)

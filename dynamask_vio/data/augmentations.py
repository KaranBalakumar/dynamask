"""
All data augmentations — visual and IMU.

Visual augmentations are applied identically to BOTH frames (prev, curr)
and to the ground truth mask (spatial transforms only).
"""

import numpy as np
import cv2
import torch


class VisualAugmentor:
    """Augments a pair of images + mask consistently."""

    def __init__(self, cfg: dict):
        aug = cfg.get("augmentation", {})
        self.color_jitter_p = aug.get("color_jitter_p", 0.8)
        self.brightness = aug.get("brightness", 0.2)
        self.contrast = aug.get("contrast", 0.2)
        self.saturation = aug.get("saturation", 0.2)
        self.hue = aug.get("hue", 0.1)
        self.flip_p = aug.get("horizontal_flip_p", 0.5)
        self.crop_p = aug.get("random_crop_p", 0.3)
        self.crop_scale = aug.get("random_crop_scale", [0.8, 1.0])
        self.noise_p = aug.get("gaussian_noise_p", 0.3)
        self.noise_sigma = aug.get("gaussian_noise_sigma", [0.0, 0.02])

    def __call__(self, img_prev: np.ndarray, img_curr: np.ndarray,
                 mask: np.ndarray, flipped: list) -> tuple:
        """
        Args:
            img_prev, img_curr: [H, W, 3] float32 in [0, 255]
            mask: [H, W] float32 in {0, 1}
            flipped: single-element list, set to True if horizontally flipped
                     (caller uses this to flip IMU axes)

        Returns:
            img_prev, img_curr, mask (all augmented)
        """
        H, W = img_prev.shape[:2]
        flipped[0] = False

        # Color jitter (same params for both frames)
        if np.random.random() < self.color_jitter_p:
            img_prev, img_curr = self._color_jitter(img_prev, img_curr)

        # Horizontal flip
        if np.random.random() < self.flip_p:
            img_prev = img_prev[:, ::-1].copy()
            img_curr = img_curr[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            flipped[0] = True

        # Random crop and resize
        if np.random.random() < self.crop_p:
            scale = np.random.uniform(*self.crop_scale)
            new_h, new_w = int(H * scale), int(W * scale)
            y0 = np.random.randint(0, H - new_h + 1)
            x0 = np.random.randint(0, W - new_w + 1)

            img_prev = cv2.resize(img_prev[y0:y0 + new_h, x0:x0 + new_w],
                                   (W, H), interpolation=cv2.INTER_LINEAR)
            img_curr = cv2.resize(img_curr[y0:y0 + new_h, x0:x0 + new_w],
                                   (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask[y0:y0 + new_h, x0:x0 + new_w],
                               (W, H), interpolation=cv2.INTER_NEAREST)

        # Gaussian noise (sigma is in [0, 255] scale now)
        if np.random.random() < self.noise_p:
            sigma = np.random.uniform(*self.noise_sigma) * 255.0
            noise = np.random.randn(*img_prev.shape).astype(np.float32) * sigma
            img_prev = np.clip(img_prev + noise, 0.0, 255.0)
            img_curr = np.clip(img_curr + noise, 0.0, 255.0)

        return img_prev, img_curr, mask

    def _color_jitter(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """Apply identical color jitter to both images (expects [0, 255] range)."""
        b = np.random.uniform(1 - self.brightness, 1 + self.brightness)
        c = np.random.uniform(1 - self.contrast, 1 + self.contrast)
        s = np.random.uniform(1 - self.saturation, 1 + self.saturation)
        h = np.random.uniform(-self.hue, self.hue)

        def apply(img):
            # Brightness
            img = img * b
            # Contrast
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - mean) * c + mean
            # Saturation
            gray = np.mean(img, axis=2, keepdims=True)
            img = gray + (img - gray) * s
            # Hue shift (convert to [0,1] for cv2 HSV, then back)
            if abs(h) > 1e-4:
                img_01 = np.clip(img / 255.0, 0, 1).astype(np.float32)
                hsv = cv2.cvtColor(img_01, cv2.COLOR_RGB2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] / 360.0 + h) % 1.0 * 360.0
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0
            return np.clip(img, 0.0, 255.0).astype(np.float32)

        return apply(img1), apply(img2)


class IMUAugmentor:
    """Augments IMU window data."""

    def __init__(self, cfg: dict):
        aug = cfg.get("augmentation", {})
        self.noise_accel = aug.get("imu_noise_accel", [0.01, 0.1])
        self.noise_gyro = aug.get("imu_noise_gyro", [0.001, 0.01])
        self.bias_drift_p = aug.get("bias_drift_p", 0.5)
        self.b0_sigma = aug.get("bias_drift_b0_sigma", 0.01)
        self.alpha_sigma = aug.get("bias_drift_alpha_sigma", 0.001)
        self.temporal_offset_p = aug.get("temporal_offset_p", 0.5)
        self.temporal_offset_ms = aug.get("temporal_offset_ms", [-10, 10])
        self.dropout_p = aug.get("imu_dropout_p", 0.3)
        self.dropout_rate = aug.get("imu_dropout_rate", 0.1)

    def __call__(self, imu_window: np.ndarray, valid_mask: np.ndarray,
                 horizontally_flipped: bool = False) -> tuple:
        """
        Args:
            imu_window: [N, 7] — (dt, ax, ay, az, gx, gy, gz)
            valid_mask: [N] boolean
            horizontally_flipped: if True, negate IMU gy and ax

        Returns:
            imu_window, valid_mask (both augmented)
        """
        imu = imu_window.copy()
        mask = valid_mask.copy()
        N = mask.sum()  # number of valid samples

        if N == 0:
            return imu, mask

        # Horizontal flip → negate gy (col 5) and ax (col 1)
        if horizontally_flipped:
            imu[:, 1] *= -1  # ax
            imu[:, 5] *= -1  # gy

        # Additive Gaussian noise (always applied to valid samples)
        sigma_a = np.random.uniform(*self.noise_accel)
        sigma_g = np.random.uniform(*self.noise_gyro)
        noise = np.zeros_like(imu[:, 1:7])
        noise[:, 0:3] = np.random.randn(imu.shape[0], 3) * sigma_a
        noise[:, 3:6] = np.random.randn(imu.shape[0], 3) * sigma_g
        imu[:, 1:7] += noise * mask[:, None].astype(np.float32)

        # Bias drift injection
        if np.random.random() < self.bias_drift_p:
            b0 = np.random.randn(6) * self.b0_sigma
            alpha = np.random.randn(6) * self.alpha_sigma
            t = imu[:, 0:1]  # relative timestamps
            drift = b0[None, :] + alpha[None, :] * t
            imu[:, 1:7] += drift * mask[:, None].astype(np.float32)

        # Temporal offset injection
        if np.random.random() < self.temporal_offset_p:
            offset_ms = np.random.uniform(*self.temporal_offset_ms)
            imu[:, 0] += offset_ms / 1000.0  # convert ms to seconds

        # Sample dropout
        if np.random.random() < self.dropout_p and N > 2:
            n_drop = max(1, int(N * self.dropout_rate))
            valid_idx = np.where(mask)[0]
            # Don't drop first or last valid sample
            if len(valid_idx) > 2:
                drop_candidates = valid_idx[1:-1]
                drop_idx = np.random.choice(drop_candidates,
                                             size=min(n_drop, len(drop_candidates)),
                                             replace=False)
                mask[drop_idx] = False
                imu[drop_idx] = 0.0

        return imu, mask

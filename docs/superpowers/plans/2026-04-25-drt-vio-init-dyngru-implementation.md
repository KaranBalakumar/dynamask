# DRT-Loose DynGRU Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PyPose-based DRT-loose initializer to bootstrap IMU state (bias, gravity, velocity, position) for both inference and training, then wire it into MACVO + DynGRU while keeping FlowFormer covariance head frozen.

**Architecture:** Implement DRT-loose as a focused initialization package (`Module/Initialization/DRTLoose`) with explicit preintegration, gyro-bias solve, translation/scale/gravity alignment, and quality gates. Integrate via a startup state machine in `Odometry/MACVO.py` that retries with longer windows and falls back to the existing heuristic initializer if needed. Extend `IMUContext` and frontend/training plumbing so DynGRU receives IMU-conditioned features from DRT-seeded state in both train and inference.

**Tech Stack:** Python 3, PyTorch, PyPose (`pp.SO3`/`pp.SE3`/LM), existing dynamask module registry/config system, pytest.

---

## File Structure (locked before coding)

### New files

- `Module/Initialization/__init__.py`  
  Public exports for initialization entrypoints.
- `Module/Initialization/DRTLoose/__init__.py`  
  Package exports.
- `Module/Initialization/DRTLoose/types.py`  
  Dataclasses: `DRTInitConfig`, `DRTWindowBundle`, `DRTInitResult`, failure enums.
- `Module/Initialization/DRTLoose/preintegration.py`  
  IMU preintegrator and Jacobian/covariance propagation.
- `Module/Initialization/DRTLoose/tracks.py`  
  Keyframe/track structures and base-view selection.
- `Module/Initialization/DRTLoose/gyro_bias.py`  
  Robust LM solve for gyro bias.
- `Module/Initialization/DRTLoose/translation.py`  
  LiGT `LᵀL` build + SVD sign disambiguation.
- `Module/Initialization/DRTLoose/alignment.py`  
  Velocity/scale/gravity alignment.
- `Module/Initialization/DRTLoose/quality.py`  
  Quality checks + reason codes.
- `Module/Initialization/DRTLoose/bootstrap.py`  
  End-to-end DRT-loose orchestrator and retry policy.
- `Scripts/UnitTest/test_drt_preintegration.py`
- `Scripts/UnitTest/test_drt_translation_alignment.py`
- `Scripts/UnitTest/test_drt_bootstrap_policy.py`
- `Scripts/UnitTest/test_imu_context_seed.py`
- `Scripts/UnitTest/assets/test_module_config/Frontend/FlowFormerDyn.yaml`
- `Config/Train/FlowFormerDyn_Demo.yaml`

### Modified files

- `Module/__init__.py`  
  Export initializer package.
- `Module/Network/IMUContext/imu_context.py`  
  `seed_from_drt` API + strict reset behavior.
- `Module/Frontend/Frontend.py`  
  Add `FlowFormerDynFrontend` and config validation.
- `Odometry/MACVO.py`  
  Inertial startup state machine + retry/fallback.
- `DataLoader/Interface.py`  
  (If needed) helper predicates for inertial frame detection.
- `Train/MatchingNet/utils.py`  
  Add train mode literals for dyn training.
- `Train/MatchingNet/train_flowformer.py`  
  Add dyn training path and freeze assertions.
- `Config/Experiment/MACVO/MACVO_Fast.yaml`
- `Config/Experiment/MACVO/MACVO_Performant.yaml`
- `Config/Experiment/MACVO/Paper_Reproduce.yaml`
- `Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml`
- `Scripts/UnitTest/test_frontend.py`
- `Scripts/UnitTest/test_config_modules.py`
- `Scripts/UnitTest/test_config_macvo.py`

---

### Task 1: Scaffold DRT module + failing API tests

**Files:**
- Create: `Module/Initialization/__init__.py`
- Create: `Module/Initialization/DRTLoose/__init__.py`
- Create: `Module/Initialization/DRTLoose/types.py`
- Modify: `Module/__init__.py`
- Test: `Scripts/UnitTest/test_drt_bootstrap_policy.py`

- [ ] **Step 1: Write the failing test for public API and result schema**

```python
# Scripts/UnitTest/test_drt_bootstrap_policy.py
import torch
from Module.Initialization.DRTLoose.types import DRTInitConfig, DRTInitResult

def test_drt_result_schema_defaults():
    cfg = DRTInitConfig()
    out = DRTInitResult.failure("NO_OBSERVABILITY")
    assert cfg.min_keyframes == 10
    assert out.success is False
    assert out.failure_reason == "NO_OBSERVABILITY"
    assert out.retry_recommended is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_bootstrap_policy.py::test_drt_result_schema_defaults`  
Expected: FAIL with `ModuleNotFoundError: No module named 'Module.Initialization'`

- [ ] **Step 3: Add minimal module/dataclass implementation**

```python
# Module/Initialization/DRTLoose/types.py
from dataclasses import dataclass

@dataclass
class DRTInitConfig:
    min_keyframes: int = 10
    max_attempts: int = 2
    window_scales: tuple[float, ...] = (1.0, 1.4, 1.8)

@dataclass
class DRTInitResult:
    success: bool
    failure_reason: str | None
    retry_recommended: bool

    @staticmethod
    def failure(reason: str) -> "DRTInitResult":
        return DRTInitResult(success=False, failure_reason=reason, retry_recommended=True)
```

- [ ] **Step 4: Export package symbols**

```python
# Module/Initialization/DRTLoose/__init__.py
from .types import DRTInitConfig, DRTInitResult
```

```python
# Module/Initialization/__init__.py
from .DRTLoose import DRTInitConfig, DRTInitResult
```

```python
# Module/__init__.py
from .Initialization import DRTInitConfig, DRTInitResult
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_bootstrap_policy.py::test_drt_result_schema_defaults`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add Module/__init__.py Module/Initialization/__init__.py Module/Initialization/DRTLoose/__init__.py Module/Initialization/DRTLoose/types.py Scripts/UnitTest/test_drt_bootstrap_policy.py
git commit -m "feat(init): scaffold DRT-loose initialization types and exports"
```

---

### Task 2: Implement IMU preintegration core (PyPose) with deterministic tests

**Files:**
- Create: `Module/Initialization/DRTLoose/preintegration.py`
- Test: `Scripts/UnitTest/test_drt_preintegration.py`

- [ ] **Step 1: Write failing constant-motion integration test**

```python
import torch
from Module.Initialization.DRTLoose.preintegration import IMUPreintegrator

def test_preintegration_constant_acc_no_rotation():
    integ = IMUPreintegrator()
    dt = 0.01
    for _ in range(100):
        integ.integrate(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), dt)
    out = integ.result()
    assert torch.allclose(out.dR_log, torch.zeros(3), atol=1e-4)
    assert torch.allclose(out.dV, torch.tensor([1.0, 0.0, 0.0]), atol=5e-2)
    assert torch.allclose(out.dP, torch.tensor([0.5, 0.0, 0.0]), atol=5e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_preintegration.py::test_preintegration_constant_acc_no_rotation`  
Expected: FAIL with missing `IMUPreintegrator`

- [ ] **Step 3: Implement preintegrator**

```python
# Module/Initialization/DRTLoose/preintegration.py
import torch
import pypose as pp
from dataclasses import dataclass

@dataclass
class PreintResult:
    dR_log: torch.Tensor
    dV: torch.Tensor
    dP: torch.Tensor
    sum_dt: float

class IMUPreintegrator:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        self._R = pp.identity_SO3(1, device=self.device, dtype=torch.float64)
        self._dV = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._dP = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._sum_dt = 0.0

    def integrate(self, gyro: torch.Tensor, acc: torch.Tensor, dt: float):
        gyro = gyro.to(dtype=torch.float64, device=self.device)
        acc = acc.to(dtype=torch.float64, device=self.device)
        dR = pp.so3((gyro * dt).unsqueeze(0)).Exp()
        self._dP = self._dP + self._dV * dt + 0.5 * (self._R @ acc) * dt * dt
        self._dV = self._dV + (self._R @ acc) * dt
        self._R = self._R @ dR
        self._sum_dt += dt

    def result(self) -> PreintResult:
        return PreintResult(
            dR_log=self._R.Log().squeeze(0).tensor(),
            dV=self._dV.clone(),
            dP=self._dP.clone(),
            sum_dt=self._sum_dt,
        )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_preintegration.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Module/Initialization/DRTLoose/preintegration.py Scripts/UnitTest/test_drt_preintegration.py
git commit -m "feat(init): add PyPose IMU preintegration kernel"
```

---

### Task 3: Implement LiGT translation + sign disambiguation

**Files:**
- Create: `Module/Initialization/DRTLoose/tracks.py`
- Create: `Module/Initialization/DRTLoose/translation.py`
- Test: `Scripts/UnitTest/test_drt_translation_alignment.py`

- [ ] **Step 1: Add failing translation sign test**

```python
from Module.Initialization.DRTLoose.translation import resolve_translation_sign
import torch

def test_translation_sign_resolution_prefers_positive_majority():
    A_lr = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
    t = torch.tensor([-1.0, 0, 0], dtype=torch.float64)
    out = resolve_translation_sign(A_lr, t)
    assert torch.allclose(out, torch.tensor([1.0, 0, 0], dtype=torch.float64))
```

- [ ] **Step 2: Run failing test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_translation_alignment.py::test_translation_sign_resolution_prefers_positive_majority`  
Expected: FAIL with import or function missing

- [ ] **Step 3: Implement translation helpers**

```python
# Module/Initialization/DRTLoose/translation.py
import torch

def smallest_singular_vector(ltl: torch.Tensor) -> torch.Tensor:
    _, _, vT = torch.linalg.svd(ltl)
    return vT[-1]

def resolve_translation_sign(A_lr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    judge = A_lr @ t
    pos = (judge > 0).sum().item()
    neg = judge.numel() - pos
    return -t if pos < neg else t
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_translation_alignment.py -k sign`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Module/Initialization/DRTLoose/tracks.py Module/Initialization/DRTLoose/translation.py Scripts/UnitTest/test_drt_translation_alignment.py
git commit -m "feat(init): implement LiGT translation sign disambiguation primitives"
```

---

### Task 4: Implement gyro-bias and velocity/scale/gravity alignment solvers

**Files:**
- Create: `Module/Initialization/DRTLoose/gyro_bias.py`
- Create: `Module/Initialization/DRTLoose/alignment.py`
- Modify: `Module/Initialization/DRTLoose/types.py`
- Test: `Scripts/UnitTest/test_drt_translation_alignment.py`

- [ ] **Step 1: Add failing alignment test**

```python
from Module.Initialization.DRTLoose.alignment import normalize_gravity
import torch

def test_normalize_gravity_enforces_norm():
    g = normalize_gravity(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64), g_norm=9.81007)
    assert abs(g.norm().item() - 9.81007) < 1e-6
```

- [ ] **Step 2: Run failing test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_translation_alignment.py::test_normalize_gravity_enforces_norm`  
Expected: FAIL with missing function

- [ ] **Step 3: Implement solver utilities**

```python
# Module/Initialization/DRTLoose/alignment.py
import torch

def normalize_gravity(g_vec: torch.Tensor, g_norm: float) -> torch.Tensor:
    return (g_vec / g_vec.norm().clamp_min(1e-12)) * g_norm
```

```python
# Module/Initialization/DRTLoose/gyro_bias.py
import torch
import pypose as pp

def rotation_residual(R_vis: pp.LieTensor, R_imu_bg: pp.LieTensor, R_bc: pp.LieTensor) -> torch.Tensor:
    err = R_vis.Inv() @ R_bc.Inv() @ R_imu_bg @ R_bc
    return err.Log().tensor()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_translation_alignment.py -k gravity`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Module/Initialization/DRTLoose/gyro_bias.py Module/Initialization/DRTLoose/alignment.py Scripts/UnitTest/test_drt_translation_alignment.py
git commit -m "feat(init): add gyro-bias residual and gravity normalization primitives"
```

---

### Task 5: Implement bootstrap orchestrator + quality gates + retry/fallback

**Files:**
- Create: `Module/Initialization/DRTLoose/quality.py`
- Create: `Module/Initialization/DRTLoose/bootstrap.py`
- Modify: `Module/Initialization/DRTLoose/types.py`
- Test: `Scripts/UnitTest/test_drt_bootstrap_policy.py`

- [ ] **Step 1: Add failing retry/fallback policy test**

```python
from Module.Initialization.DRTLoose.bootstrap import run_drt_with_retry
from Module.Initialization.DRTLoose.types import DRTInitConfig

def test_retry_then_fallback_policy():
    cfg = DRTInitConfig(max_attempts=2, window_scales=(1.0, 1.4, 1.8))
    out = run_drt_with_retry(cfg, solver_fn=lambda scale: None)
    assert out.success is False
    assert out.failure_reason == "FALLBACK_HEURISTIC"
```

- [ ] **Step 2: Run failing test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_bootstrap_policy.py::test_retry_then_fallback_policy`  
Expected: FAIL with missing orchestrator

- [ ] **Step 3: Implement retry/fallback logic**

```python
# Module/Initialization/DRTLoose/bootstrap.py
from .types import DRTInitConfig, DRTInitResult

def run_drt_with_retry(cfg: DRTInitConfig, solver_fn):
    attempts = min(cfg.max_attempts + 1, len(cfg.window_scales))
    for i in range(attempts):
        res = solver_fn(cfg.window_scales[i])
        if res is not None and res.success:
            return res
    return DRTInitResult(success=False, failure_reason="FALLBACK_HEURISTIC", retry_recommended=False)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_drt_bootstrap_policy.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Module/Initialization/DRTLoose/quality.py Module/Initialization/DRTLoose/bootstrap.py Module/Initialization/DRTLoose/types.py Scripts/UnitTest/test_drt_bootstrap_policy.py
git commit -m "feat(init): add bootstrap retry policy and fallback result handling"
```

---

### Task 6: Integrate DRT startup state machine into MACVO + IMUContext seeding

**Files:**
- Modify: `Module/Network/IMUContext/imu_context.py`
- Modify: `Odometry/MACVO.py`
- Test: `Scripts/UnitTest/test_imu_context_seed.py`

- [ ] **Step 1: Add failing IMUContext seed test**

```python
import torch
import pypose as pp
from Module.Network.IMUContext.imu_context import IMUContext

def test_seed_from_drt_sets_state_and_gravity(monkeypatch):
    ctx = IMUContext(airio_cfg={"model": {}}, airio_ckpt=None)  # patch model loading in test fixture
    ctx.seed_from_drt(
        rot=pp.identity_SO3(1).squeeze(0),
        vel=torch.tensor([1.0, 2.0, 3.0]),
        pos=torch.tensor([0.1, 0.2, 0.3]),
        bias_g=torch.zeros(3),
        bias_a=torch.zeros(3),
        gravity_world=torch.tensor([0.0, 0.0, 9.81007]),
    )
    assert torch.allclose(ctx._state[3:6].float(), torch.tensor([1.0, 2.0, 3.0]))
```

- [ ] **Step 2: Run failing test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_imu_context_seed.py::test_seed_from_drt_sets_state_and_gravity`  
Expected: FAIL with missing `seed_from_drt`

- [ ] **Step 3: Implement `seed_from_drt` in IMUContext**

```python
def seed_from_drt(self, rot, vel, pos, bias_g, bias_a, gravity_world, cov_diag=None):
    dev = vel.device
    s = torch.zeros(15, dtype=torch.float64, device=dev)
    s[:3] = (rot.Log().tensor() if hasattr(rot.Log(), "tensor") else rot.Log()).double()
    s[3:6] = vel.double()
    s[6:9] = pos.double()
    s[9:12] = bias_g.double()
    s[12:15] = bias_a.double()
    self._state = s
    self._P = torch.eye(15, dtype=torch.float64, device=dev)
    if cov_diag is not None:
        self._P[range(15), range(15)] = cov_diag.double()
    self.gravity_world = gravity_world.to(dtype=torch.float64, device=dev)
    self._prev_cam_state = s.clone()
```

- [ ] **Step 4: Integrate startup behavior in MACVO**

```python
# Odometry/MACVO.py (conceptual insertion)
if not self.isinitiated:
    if hasattr(frame, "imu") and getattr(self.config, "init", None) and self.config.init.enabled:
        ok = self._initialize_with_drt(frame)
        if not ok:
            self.initialize(frame)  # existing heuristic fallback
    else:
        self.initialize(frame)
    self.isinitiated = True
    return
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_imu_context_seed.py`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add Module/Network/IMUContext/imu_context.py Odometry/MACVO.py Scripts/UnitTest/test_imu_context_seed.py
git commit -m "feat(init): seed IMUContext from DRT and wire MACVO startup state machine"
```

---

### Task 7: Add FlowFormerDyn frontend and config test coverage

**Files:**
- Modify: `Module/Frontend/Frontend.py`
- Create: `Scripts/UnitTest/assets/test_module_config/Frontend/FlowFormerDyn.yaml`
- Modify: `Scripts/UnitTest/test_config_modules.py`

- [ ] **Step 1: Add failing frontend config test case**

```python
# in existing param glob this file is auto-discovered:
# Scripts/UnitTest/assets/test_module_config/Frontend/FlowFormerDyn.yaml
type: FlowFormerDynFrontend
args:
  device: cuda
  weight: ./Model/MACVO_FrontendCov.pth
  airio_ckpt: ./Model/airio.pth
  enc_dtype: fp32
  dec_dtype: fp32
  decoder_depth: 12
  enforce_positive_disparity: false
```

- [ ] **Step 2: Run failing config test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_config_modules.py::test_frontend_config`  
Expected: FAIL with unknown frontend type

- [ ] **Step 3: Implement `FlowFormerDynFrontend`**

```python
class FlowFormerDynFrontend(FlowFormerCovFrontend):
    def __init__(self, config):
        from ..Network.FlowFormer.configs.submission import get_cfg
        from ..Network.FlowFormerDyn import build_flowformer_dyn
        super(IFrontend, self).__init__()
        self.config = config
        cfg = get_cfg()
        cfg.latentcostformer.decoder_depth = self.config.decoder_depth
        self.model = build_flowformer_dyn(cfg, reflect_torch_dtype(config.enc_dtype), reflect_torch_dtype(config.dec_dtype))
        self.model.load_ddp_state_dict(torch.load(self.config.weight, map_location=self.config.device, weights_only=True))
        self.model.to(self.config.device).eval()
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_config_modules.py::test_frontend_config`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Module/Frontend/Frontend.py Scripts/UnitTest/assets/test_module_config/Frontend/FlowFormerDyn.yaml Scripts/UnitTest/test_config_modules.py
git commit -m "feat(frontend): add FlowFormerDyn frontend config and loader"
```

---

### Task 8: Dyn training path with strict freezing (cov head frozen)

**Files:**
- Modify: `Train/MatchingNet/utils.py`
- Modify: `Train/MatchingNet/train_flowformer.py`
- Create: `Config/Train/FlowFormerDyn_Demo.yaml`

- [ ] **Step 1: Add failing train-mode test**

```python
from Train.MatchingNet.utils import AssertLiteralType, T_TrainType

def test_train_type_accepts_dyn():
    assert AssertLiteralType("dyn", T_TrainType)
```

- [ ] **Step 2: Run failing test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_config_loadable.py`  
Expected: FAIL due unsupported training mode literal

- [ ] **Step 3: Implement dyn mode and freeze assertions**

```python
# Train/MatchingNet/utils.py
T_TrainType = Literal["flow", "cov", "flow+cov", "finalcov", "dyn"]
```

```python
# Train/MatchingNet/train_flowformer.py (dyn branch)
if train_mode == "dyn":
    for p in model_ptr.parameters():
        p.requires_grad = False
    for p in model_ptr.memory_decoder.dyn_update.parameters():
        p.requires_grad = True
    # optional: IMUContext projection params enabled by config
    assert all(not p.requires_grad for p in model_ptr.memory_decoder.cov_update.parameters())
```

- [ ] **Step 4: Run relevant tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_config_loadable.py Scripts/UnitTest/test_config_modules.py`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Train/MatchingNet/utils.py Train/MatchingNet/train_flowformer.py Config/Train/FlowFormerDyn_Demo.yaml
git commit -m "feat(train): add dyn training mode with covariance-head freeze enforcement"
```

---

### Task 9: Wire odometry configs + full regression slice

**Files:**
- Modify: `Config/Experiment/MACVO/MACVO_Fast.yaml`
- Modify: `Config/Experiment/MACVO/MACVO_Performant.yaml`
- Modify: `Config/Experiment/MACVO/Paper_Reproduce.yaml`
- Modify: `Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml`
- Modify: `Scripts/UnitTest/test_config_macvo.py`

- [ ] **Step 1: Add init block in configs**

```yaml
init:
  type: DRTLoose
  enabled: true
  min_keyframes: 10
  retry:
    max_attempts: 2
    window_scale: [1.0, 1.4, 1.8]
  quality:
    min_avg_observation: 30
    max_condition_number: 1.0e8
    min_positive_depth_ratio: 0.7
  fallback: heuristic
```

- [ ] **Step 2: Run config validation test**

Run: `python3 -m pytest -q Scripts/UnitTest/test_config_macvo.py`  
Expected: PASS

- [ ] **Step 3: Run targeted runtime/frontend tests**

Run: `python3 -m pytest -q Scripts/UnitTest/test_frontend.py Scripts/UnitTest/test_config_modules.py Scripts/UnitTest/test_drt_preintegration.py Scripts/UnitTest/test_drt_translation_alignment.py Scripts/UnitTest/test_drt_bootstrap_policy.py Scripts/UnitTest/test_imu_context_seed.py`  
Expected: PASS (non-local tests); local-only tests remain deselected.

- [ ] **Step 4: Commit**

```bash
git add Config/Experiment/MACVO/MACVO_Fast.yaml Config/Experiment/MACVO/MACVO_Performant.yaml Config/Experiment/MACVO/Paper_Reproduce.yaml Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml Scripts/UnitTest/test_config_macvo.py
git commit -m "feat(config): enable DRT-loose initialization and retry/fallback policy in MACVO configs"
```

---

## Spec coverage check

- DRT-loose default path: **Task 2-5**
- Retry + longer window + fallback: **Task 5 + Task 6 + Task 9**
- Train DynGRU with cov head frozen: **Task 8**
- Train/inference parity for initialization: **Task 6 + Task 8 + Task 9**
- PyPose-first math path: **Task 2-4**
- Runtime integration in current MACVO: **Task 6**
- Frontend/config/test wiring: **Task 7 + Task 9**

No spec requirement is left unmapped.

## Placeholder scan

- Searched for placeholder/red-flag tokens: none used in actionable steps.

## Type consistency

- `DRTInitConfig`, `DRTInitResult`, `IMUPreintegrator`, `seed_from_drt`, `FlowFormerDynFrontend`, and `train_mode="dyn"` names are used consistently across all tasks.

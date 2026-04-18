from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pypose as pp
import torch

from Module.Map import FrameNode, VisualMap
from Module.Optimization.TwoFramePGO.Graphs import GraphOutput
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Module.Optimization.SlidingWindow.Optimizer import SlidingWindow_VIO_PGO, SWGraphOutput


def _build_map_with_one_frame() -> VisualMap:
    gmap = VisualMap()
    gmap.frames.push(
        FrameNode.init(
            {
                "pose": pp.identity_SE3(1).tensor().float(),
                "T_BS": pp.identity_SE3(1).tensor().float(),
                "vel": torch.zeros((1, 3), dtype=torch.float32),
                "bias_g": torch.zeros((1, 3), dtype=torch.float32),
                "bias_a": torch.zeros((1, 3), dtype=torch.float32),
                "need_interp": torch.tensor([0], dtype=torch.bool),
                "time_ns": torch.tensor([0], dtype=torch.long),
                "K": torch.eye(3, dtype=torch.float32).unsqueeze(0),
                "baseline": torch.tensor([0.1], dtype=torch.float32),
            }
        )
    )
    return gmap


def test_twoframe_writeback_emits_logging():
    cfg = SimpleNamespace(graph_type="reproj", device="cpu", vectorize=True, parallel=False, autodiff=True)
    optimizer = TwoFrame_PGO(cfg)
    logger = Mock()
    optimizer.debug_logger = logger
    gmap = _build_map_with_one_frame()
    result = GraphOutput(
        motion=pp.identity_SE3(1),
        from_idx=torch.tensor([0], dtype=torch.long),
        frame_idx=torch.tensor([0], dtype=torch.long),
        vel=torch.zeros(3, dtype=torch.float64),
        bias_g=torch.zeros(3, dtype=torch.float64),
        bias_a=torch.zeros(3, dtype=torch.float64),
        diagnostics={"backend.twoframe.chi2.init": 1.0},
        dump_payload={"chi2_init": torch.tensor(1.0)},
    )
    optimizer.write_graph_data(result, gmap)
    assert logger.log_scalars.call_count >= 2
    logger.dump_artifact.assert_called_once()


def test_slidingwindow_writeback_emits_logging():
    cfg = SimpleNamespace(graph_type="reproj", device="cpu", vectorize=True, parallel=False, autodiff=True, window_size=2)
    optimizer = SlidingWindow_VIO_PGO(cfg)
    logger = Mock()
    optimizer.debug_logger = logger
    gmap = _build_map_with_one_frame()
    pair_out = GraphOutput(
        motion=pp.identity_SE3(1),
        from_idx=torch.tensor([0], dtype=torch.long),
        frame_idx=torch.tensor([0], dtype=torch.long),
        vel=torch.zeros(3, dtype=torch.float64),
        bias_g=torch.zeros(3, dtype=torch.float64),
        bias_a=torch.zeros(3, dtype=torch.float64),
    )
    result = SWGraphOutput(
        results=[pair_out],
        diagnostics={"backend.swf.chi2.init": 1.0},
        dump_payload={"chi2_init": torch.tensor(1.0)},
    )
    optimizer.write_graph_data(result, gmap)
    assert logger.log_scalars.call_count >= 2
    logger.dump_artifact.assert_called_once()

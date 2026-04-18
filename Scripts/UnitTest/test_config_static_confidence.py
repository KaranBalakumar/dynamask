from pathlib import Path

from Odometry.MACVO import MACVO
from Utility.Config import load_config


def test_static_confidence_config_loadable():
    cfg, _ = load_config(Path("./Config/Experiment/StaticConfidenceHead/viode.yaml"))
    MACVO.is_valid_config(cfg.Odometry)


---
type: "query"
date: "2026-04-14T19:47:09.610854+00:00"
question: "explain MAC-VO"
contributor: "graphify"
source_nodes: ["MACVO"]
---

# Q: explain MAC-VO

## Answer

The current corrected graph scope is dynamask_vio/models, so it has no MAC-VO node. From repository sources, MAC-VO is Metrics-aware Covariance for Learning-based Stereo Visual Odometry, with a modular interface-driven architecture. The core MACVO class in references/MAC-VO/Odometry/MACVO.py orchestrates Frontend, MotionModel, KeypointSelector, OutlierFilter, covariance model, keyframe selector, map processor, and optimizer over a VisualMap graph. Config/README.md shows modules are dynamically instantiated from YAML interfaces, enabling swappable implementations without changing orchestration logic.

## Source Nodes

- MACVO
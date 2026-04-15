---
type: "query"
date: "2026-04-14T19:36:47.502575+00:00"
question: "Cleaned dependency view for dynamask_vio/models bridge slice"
contributor: "graphify"
source_nodes: ["DynaMaskVIO", "IMUEncoder", "BasicEncoder", "FlowDecoder", "ScoreHead", "FiLM"]
---

# Q: Cleaned dependency view for dynamask_vio/models bridge slice

## Answer

Trusted dependency view from source code: DynaMaskVIO imports and instantiates IMUEncoder, BasicEncoder, and FlowDecoder in dynamask.py lines 9-12 and 31-65. DynaMaskVIO forward calls IMUEncoder at line 87, feeds f_imu to BasicEncoder at lines 93-94, and calls FlowDecoder at lines 100-102. FlowDecoder imports ScoreHead at flow_decoder.py line 23, instantiates it at lines 259-263, and calls it at line 310. BasicEncoder applies FiLM conditioning with f_imu at backbone.py lines 165-176, and FiLM computes gamma and beta modulation in film.py lines 46-48. ScoreHead uses its local GradientClip in score_head.py lines 21 and 38. Corrected bridge directions are DynaMaskVIO to IMUEncoder, DynaMaskVIO to BasicEncoder, DynaMaskVIO to FlowDecoder, and FlowDecoder to ScoreHead.

## Source Nodes

- DynaMaskVIO
- IMUEncoder
- BasicEncoder
- FlowDecoder
- ScoreHead
- FiLM
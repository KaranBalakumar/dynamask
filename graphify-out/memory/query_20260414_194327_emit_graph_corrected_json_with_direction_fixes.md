---
type: "query"
date: "2026-04-14T19:43:27.481868+00:00"
question: "Emit graph.corrected.json with direction fixes"
contributor: "graphify"
source_nodes: ["DynaMaskVIO", "IMUEncoder", "FlowDecoder", "ScoreHead", "BasicEncoder"]
---

# Q: Emit graph.corrected.json with direction fixes

## Answer

Created graph.corrected.json from graph.filtered.json with direction-corrected extracted call edges: DynaMaskVIO to IMUEncoder, DynaMaskVIO to FlowDecoder, FlowDecoder to ScoreHead, and DynaMaskVIO to BasicEncoder. Two inferred reverse-direction edges were removed. The corrected graph has 147 nodes and 196 edges with zero inferred ScoreHead edges. Matching outputs GRAPH_REPORT.corrected.md and graph.corrected.html were generated.

## Source Nodes

- DynaMaskVIO
- IMUEncoder
- FlowDecoder
- ScoreHead
- BasicEncoder
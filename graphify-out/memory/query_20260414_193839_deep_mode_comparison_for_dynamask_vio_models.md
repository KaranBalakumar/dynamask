---
type: "query"
date: "2026-04-14T19:38:39.918901+00:00"
question: "Deep mode comparison for dynamask_vio/models"
contributor: "graphify"
source_nodes: ["ScoreHead", "FlowDecoder", "DynaMaskVIO", "BasicEncoder", "IMUEncoder"]
---

# Q: Deep mode comparison for dynamask_vio/models

## Answer

Deep mode produced no change for this corpus because it is code-only, so semantic extraction is skipped. Before and after counts are identical at 147 nodes and 213 edges. ScoreHead inferred edges remain 18 with the exact same target set. Conclusion: deep mode neither improved nor worsened bridge-edge quality on this slice.

## Source Nodes

- ScoreHead
- FlowDecoder
- DynaMaskVIO
- BasicEncoder
- IMUEncoder
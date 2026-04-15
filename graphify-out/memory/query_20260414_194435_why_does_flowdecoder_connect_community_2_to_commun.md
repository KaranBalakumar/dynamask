---
type: "query"
date: "2026-04-14T19:44:35.046822+00:00"
question: "Why does FlowDecoder connect Community 2 to Community 8, Community 1, Community 5 in corrected graph?"
contributor: "graphify"
source_nodes: ["FlowDecoder", "ScoreHead", "CorrBlock", "MotionEncoder", "FlowHead", "DynaMaskVIO", "BasicEncoder"]
---

# Q: Why does FlowDecoder connect Community 2 to Community 8, Community 1, Community 5 in corrected graph?

## Answer

FlowDecoder bridges those three communities through explicit structural links. It connects to Community 5 via a direct extracted call edge to ScoreHead at flow_decoder.py line 310. It connects to Community 1 through extracted contains and method links into flow_decoder.py internals such as MotionEncoder and FlowHead. It connects to Community 8 via extracted forward and CorrBlock-related structure from flow_decoder.py. Community 2 ties come from model wiring with DynaMaskVIO and BasicEncoder. This bridge is now primarily explicit after edge cleanup and correction.

## Source Nodes

- FlowDecoder
- ScoreHead
- CorrBlock
- MotionEncoder
- FlowHead
- DynaMaskVIO
- BasicEncoder
---
type: "query"
date: "2026-04-14T19:40:30.283009+00:00"
question: "Why does FlowDecoder connect Community 2 to Community 8 and Community 1 in filtered graph?"
contributor: "graphify"
source_nodes: ["FlowDecoder", "DynaMaskVIO", "BasicEncoder", "CorrBlock", "MotionEncoder", "FlowHead"]
---

# Q: Why does FlowDecoder connect Community 2 to Community 8 and Community 1 in filtered graph?

## Answer

FlowDecoder bridges those communities through mixed structural and inferred links. Structural links: FlowDecoder contains flow_decoder.py and methods, which connect it to MotionEncoder and FlowHead in Community 1 and to CorrBlock and forward-related nodes in Community 8. Inferred links place FlowDecoder near DynaMaskVIO and then BasicEncoder in Community 2, creating a path toward RAFT-encoding nodes. The Community 2 bridge is partially directionally noisy: code shows DynaMaskVIO instantiates and calls FlowDecoder, not the reverse. So Community 1 and 8 links are strongly structural, while the Community 2 link is semantically meaningful but inference-direction sensitive.

## Source Nodes

- FlowDecoder
- DynaMaskVIO
- BasicEncoder
- CorrBlock
- MotionEncoder
- FlowHead